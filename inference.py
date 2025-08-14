# load packages
import random
import yaml
import time
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
import click
import shutil
import warnings
import torchaudio
import torch
import soundfile as sf
import json
warnings.simplefilter('ignore')
from torch.utils.tensorboard import SummaryWriter
from phonemizer import phonemize
from phonemizer.separator import Separator
from meldataset import build_dataloader
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert

from models import *
from losses import *
from utils import *

from Modules.slmadv import SLMAdversarialLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from optimizers import build_optimizer

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
import logging
from logging import StreamHandler
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
@click.option('--inference', is_flag=True, help="Run inference mode")
@click.option('--audio_path', type=str, help="Path to input audio for inference.")
@click.option('--text', type=str, help="Input text for inference.")
def main(config_path, inference, audio_path, text):
    config = yaml.safe_load(open(config_path))
    
    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.addHandler(file_handler)

    epochs = config.get('epochs', 200)
    data_params = config.get('data_params', None)
    sr = config['preprocess_params'].get('sr', 24000)
    root_path = data_params['root_path']
    min_length = data_params['min_length']
    max_len = config.get('max_len', 200)
    
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    
    optimizer_params = Munch(config['optimizer_params'])

    # --- Parameters inf ---
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
    mean, std = -4, 4

    # --- Utility functions inf ---
    def extract_mel(wave):
        # wave: numpy array (T,)
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor  # shape: (1, 80, T)
        
    def prepare_audio(audio_path, sample_rate=24000):
        wav, sr = sf.read(audio_path)
        if len(wav.shape) == 2:
            wav = wav[:, 0]
        if sr != sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=sample_rate)
        # Pad with 5000 zeros at start and end (as in meldataset.py)
        wav = np.concatenate([np.zeros([5000]), wav, np.zeros([5000])], axis=0)
        return wav

    device = 'cuda'
    
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # load PL-BERT model
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # build model
    model_params = recursive_munch(config['model_params'])
    multispeaker = model_params.multispeaker
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]
    
    # DP
    for key in model:
        if key != "mpd" and key != "msd" and key != "wd":
            model[key] = MyDataParallel(model[key])
            
    start_epoch = 0
    iters = 0

    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print('Loading the first stage model at %s ...' % first_stage_path)
            model, _, start_epoch, iters = load_checkpoint(model, 
                None, 
                first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']) # keep starting epoch for tensorboard log

            # these epochs should be counted from the start epoch
            diff_epoch += start_epoch
            joint_epoch += start_epoch
            epochs += start_epoch
            
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError('You need to specify the path to the first stage model.') 

    gl = GeneratorLoss(model.mpd, model.msd).to(device)
    dl = DiscriminatorLoss(model.mpd, model.msd).to(device)
    wl = WavLMLoss(model_params.slm.model, 
                   model.wd, 
                   sr, 
                   model_params.slm.sr).to(device)

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
        clamp=False
    )
    
    scheduler_params = {
        "max_lr": optimizer_params.lr,
        "pct_start": float(0),
        
    }
  
    scheduler_params_dict= {key: scheduler_params.copy() for key in model}
    scheduler_params_dict['bert']['max_lr'] = optimizer_params.bert_lr * 2
    scheduler_params_dict['decoder']['max_lr'] = optimizer_params.ft_lr * 2
    scheduler_params_dict['style_encoder']['max_lr'] = optimizer_params.ft_lr * 2
    
    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                          scheduler_params_dict=scheduler_params_dict, lr=optimizer_params.lr)
    
    # adjust BERT learning rate
    for g in optimizer.optimizers['bert'].param_groups:
        g['betas'] = (0.9, 0.99)
        g['lr'] = optimizer_params.bert_lr
        g['initial_lr'] = optimizer_params.bert_lr
        g['min_lr'] = 0
        g['weight_decay'] = 0.01
        
    # adjust acoustic module learning rate
    for module in ["decoder", "style_encoder"]:
        for g in optimizer.optimizers[module].param_groups:
            g['betas'] = (0.0, 0.99)
            g['lr'] = optimizer_params.ft_lr
            g['initial_lr'] = optimizer_params.ft_lr
            g['min_lr'] = 0
            g['weight_decay'] = 1e-4
        
    # load models if there is a model
    if load_pretrained:
        model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))
        
    n_down = model.text_aligner.n_down

    def inference_viseme_json(audio_path, text):
        """
        Given an audio file and input text, produce viseme JSON using forced alignment from the ASR model.
        Excludes certain phonemes from being assigned to viseme frames, but still advances time for them.
        """
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        import json
        import torch
    
        # Load phoneme_to_viseme mapping
        with open('/content/phoneme_to_viseme.json', 'r', encoding='utf-8') as f:
            phoneme_to_viseme = json.load(f)
    
        # Skip symbols (from training debug patch)
        SKIPPED_SYMBOLS = set(';:,.!?¡¿—…\"«»“”ǃːˈˌˑʼ˞↓↑→↗↘̩')
    
        hop_length = 300
        sample_rate = 24000
        frame_duration_ms = 25.0  
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        # 1. Load and preprocess audio to get mel spectrogram
        wav = prepare_audio(audio_path, sample_rate=sample_rate)
        mels = extract_mel(wav).to(device)
    
        # 2. Phonemize the input text
        phoneme_text = phonemize(
            text,
            language='en-us',
            backend='espeak',
            strip=False,
            preserve_punctuation=False,
            njobs=1,
            separator=Separator(phone=' ', word=' | ')
        )
    
        print("Phonemized text:", phoneme_text)
        phonemes = phoneme_text.strip().split()
        phonemes = [p for p in phonemes if p != '|']
        print("Phoneme tokens:", phonemes)
    
        # 3. Prepare text ids
        from meldataset import TextCleaner
        text_cleaner = TextCleaner()
        text_ids = torch.LongTensor(
            text_cleaner(" ".join(phonemes))
        ).unsqueeze(0).to(device)
        input_lengths = torch.LongTensor([text_ids.shape[1]]).to(device)
    
        # 4. Run forced alignment
        text_aligner = model['text_aligner']
        text_aligner.eval()
        n_down = text_aligner.n_down
    
        with torch.no_grad():
            mel_len = mels.shape[-1]
            mask_length = (mel_len + (2 ** n_down) - 1) // (2 ** n_down)
            mask = length_to_mask(torch.LongTensor([mask_length])).to(device)
            _, _, s2s_attn = text_aligner(mels, mask, text_ids)
    
            s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
            mask_ST = mask_from_lens(
                s2s_attn,
                input_lengths,
                torch.LongTensor([mel_len // (2 ** n_down)]).to(device)
            )
            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
            d_gt = s2s_attn_mono.sum(axis=-1).detach().cpu().numpy()
            ph_ids = text_ids[0].cpu().tolist()
            durations = d_gt[0].tolist()
    
        id2ph = {v: k for k, v in text_cleaner.word_index_dictionary.items()}
    
        # 5. Build viseme JSON 
        output_json = []
        start_ms = 0.0
        for ph_id, dur in zip(ph_ids, durations):
            symbol = id2ph.get(ph_id, f"[UNK_{ph_id}]")
            duration_ms = dur * frame_duration_ms
    
            if symbol in SKIPPED_SYMBOLS:
                print(f"[INFO] Skipping viseme mapping for '{symbol}', advancing time by {duration_ms:.2f} ms")
                start_ms += duration_ms
                continue
    
            viseme_id = phoneme_to_viseme.get(symbol, 0)
            if symbol not in phoneme_to_viseme:
                print(f"[INFO] Phoneme '{symbol}' does not have a viseme mapping .")
    
            output_json.append({
                "offset": round(start_ms, 3),
                "visemeId": viseme_id
            })
            start_ms += duration_ms
    
        print(json.dumps(output_json, indent=4))
        return output_json
        
    # inference function call  
    if inference:
        assert audio_path is not None, "You must provide --audio_path for inference"
        assert text is not None, "You must provide --text for inference"
        inference_viseme_json(audio_path, text)
        return
                            
if __name__=="__main__":
    main()
