import os.path as osp
import yaml
import copy
import torch
import shutil
from munch import Munch
from torch.utils.tensorboard import SummaryWriter
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Utils.PLBERT.util import load_plbert
from models import build_model
from utils import load_checkpoint, recursive_munch, load_ASR_models, load_F0_models
from losses import GeneratorLoss, DiscriminatorLoss, WavLMLoss
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from optimizers import build_optimizer
import os
import os.path as osp
import json
import numpy as np
import torch
import soundfile as sf
import librosa
import torchaudio
from phonemizer import phonemize
from phonemizer.separator import Separato
from meldataset import TextCleaner
from utils import length_to_mask, mask_from_lens, maximum_path


# DataParallel fix
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def load_model(config_path="Configs/config_ft.yml", device="cuda"):
    """
    Loads all pretrained models, configs, optimizer, and returns model objects
    for inference or training.
    """
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    writer = SummaryWriter(log_dir + "/tensorboard")

    # Parameters
    loss_params = Munch(config['loss_params'])
    diff_epoch = loss_params.diff_epoch
    joint_epoch = loss_params.joint_epoch
    optimizer_params = Munch(config['optimizer_params'])

    # Load pretrained ASR model
    text_aligner = load_ASR_models(config.get('ASR_path'), config.get('ASR_config'))
    # Load F0 extractor
    pitch_extractor = load_F0_models(config.get('F0_path'))
    # Load PL-BERT
    plbert = load_plbert(config.get('PLBERT_dir'))

    # Build model
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].to(device) for key in model]

    # Wrap in DataParallel
    for key in model:
        if key not in ["mpd", "msd", "wd"]:
            model[key] = MyDataParallel(model[key])

    # Handle pretrained checkpoints
    load_pretrained = config.get('pretrained_model', '') != '' and config.get('second_stage_load_pretrained', False)
    if not load_pretrained:
        if config.get('first_stage_path', '') != '':
            first_stage_path = osp.join(log_dir, config.get('first_stage_path', 'first_stage.pth'))
            print(f"Loading first stage model from {first_stage_path}...")
            model, _, start_epoch, iters = load_checkpoint(
                model, None, first_stage_path,
                load_only_params=True,
                ignore_modules=['bert', 'bert_encoder', 'predictor', 'predictor_encoder', 'msd', 'mpd', 'wd', 'diffusion']
            )
            model.predictor_encoder = copy.deepcopy(model.style_encoder)
        else:
            raise ValueError("You need to specify the path to the first stage model.")
    else:
        model, optimizer, start_epoch, iters = load_checkpoint(
            model, None, config['pretrained_model'],
            load_only_params=config.get('load_only_params', True)
        )

    # Losses
    gl = MyDataParallel(GeneratorLoss(model.mpd, model.msd).to(device))
    dl = MyDataParallel(DiscriminatorLoss(model.mpd, model.msd).to(device))
    wl = MyDataParallel(WavLMLoss(model_params.slm.model, model.wd, config['preprocess_params']['sr'], model_params.slm.sr).to(device))

    # Sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )

    print("[INFO] Model and components successfully loaded.")
    return {
        "model": model,
        "text_aligner": text_aligner,
        "pitch_extractor": pitch_extractor,
        "plbert": plbert,
        "losses": {"gen": gl, "disc": dl, "wavlm": wl},
        "sampler": sampler,
        "config": config,
        "device": device
    }

# ---------- load phoneme to viseme map ----------
import json
import os

with open(os.path.join(os.path.dirname(__file__), "phoneme_viseme.json"), "r") as f:
    phoneme_viseme = json.load(f)


def inference_viseme_json(
    model_bundle: dict,
    audio_path: str,
    text: str,
    mapping_path: str = "phoneme_viseme.json",
):
    """
    Run viseme inference using preloaded models.

    Args:
        model_bundle: dict returned by load_model(), containing:
            - "model" (dict of torch.nn.Modules; includes 'text_aligner')
            - "device" (e.g., 'cuda' or 'cpu')
            - "config" (loaded YAML config)
        audio_path: path to input audio (.wav or .mp3)
        text: input text (grapheme string)
        mapping_path: path to phoneme->viseme JSON mapping (default in CWD)

    Returns:
        List[dict]: [{"offset": <ms>, "visemeId": <int>}, ...]
    """

    # ---------- Resolve device & components ----------
    device = model_bundle.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model_bundle["model"]
    # Prefer the aligner inside the built model; fall back to separate return if needed
    text_aligner = model.get("text_aligner", model_bundle.get("text_aligner", None))
    if text_aligner is None:
        raise RuntimeError("text_aligner not found in model bundle.")

    # ---------- Load phoneme->viseme mapping ----------
    phoneme_viseme = {}
    tried_paths = []
    for cand in [
        mapping_path,
        osp.join(os.getcwd(), mapping_path),
        # If running as a module, __file__ may exist:
        osp.join(osp.dirname(__file__), mapping_path) if "__file__" in globals() else None,
    ]:
        if cand and osp.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                phoneme_viseme = json.load(f)
            break
        if cand:
            tried_paths.append(cand)
    if not phoneme_viseme:
        print(f"[WARN] phoneme_viseme mapping not found. Tried: {tried_paths}. "
              f"Proceeding with default mapping to 0.")

    # ---------- Constants (match your original script) ----------
    sample_rate = 24000
    hop_length = 300
    frame_duration_ms = 25.0

    # ---------- Audio loader (handles wav + mp3 robustly) ----------
    def _load_audio(audio_path_, target_sr=sample_rate):
        # soundfile has limited mp3 support; use librosa for mp3 or fallback on error
        use_librosa = audio_path_.lower().endswith(".mp3")
        if not use_librosa:
            try:
                wav, sr_ = sf.read(audio_path_)
                if wav.ndim == 2:
                    wav = wav[:, 0]
                if sr_ != target_sr:
                    wav = librosa.resample(wav, orig_sr=sr_, target_sr=target_sr)
                # pad like meldataset.py
                wav = np.concatenate([np.zeros([5000]), wav, np.zeros([5000])], axis=0)
                return wav
            except Exception as e:
                print(f"[INFO] soundfile read failed ({e}); falling back to librosa.")
                use_librosa = True
        # librosa path
        wav, sr_ = librosa.load(audio_path_, sr=target_sr, mono=True)
        wav = np.concatenate([np.zeros([5000]), wav, np.zeros([5000])], axis=0)
        return wav

    # ---------- Mel extractor (same params as your script) ----------
    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=80, n_fft=2048, win_length=1200, hop_length=hop_length
    )
    mean, std = -4, 4

    def _extract_mel(wave_np):
        wave_tensor = torch.from_numpy(wave_np).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor  # (1, 80, T)

    # ---------- Load & preprocess audio ----------
    wav = _load_audio(audio_path, target_sr=sample_rate)
    mels = _extract_mel(wav).to(device)

    # ---------- Phonemize input text ----------
    phoneme_text = phonemize(
        text,
        language="en-us",
        backend="espeak",
        strip=False,
        preserve_punctuation=False,
        njobs=1,
        separator=Separator(phone=" ", word=" | "),
    )
    phonemes = [p for p in phoneme_text.strip().split() if p != "|"]

    # ---------- Convert phonemes -> ids ----------
    text_cleaner = TextCleaner()
    text_ids = torch.LongTensor(text_cleaner(" ".join(phonemes))).unsqueeze(0).to(device)
    input_lengths = torch.LongTensor([text_ids.shape[1]]).to(device)

    # ---------- Forced alignment using text_aligner ----------
    text_aligner.eval()
    n_down = text_aligner.n_down  # downsample factor inside the aligner

    with torch.no_grad():
        mel_len = mels.shape[-1]
        # mask length in downsampled frames
        mask_length = (mel_len + (2 ** n_down) - 1) // (2 ** n_down)
        mask = length_to_mask(torch.LongTensor([mask_length])).to(device)

        # Forward pass
        _, _, s2s_attn = text_aligner(mels, mask, text_ids)

        # Convert attn to monotonic path
        s2s_attn = s2s_attn.transpose(-1, -2)[..., 1:].transpose(-1, -2)
        mask_ST = mask_from_lens(
            s2s_attn,
            input_lengths,
            torch.LongTensor([mel_len // (2 ** n_down)]).to(device),
        )
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # Durations per phoneme (in downsampled mel frames)
        d_gt = s2s_attn_mono.sum(axis=-1).detach().cpu().numpy()
        ph_ids = text_ids[0].detach().cpu().tolist()
        durations = d_gt[0].tolist()

    # ---------- Map phoneme ids back to symbols ----------
    id2ph = {v: k for k, v in text_cleaner.word_index_dictionary.items()}

    # ---------- Build viseme JSON ----------
    # Skip symbols 
    SKIPPED_SYMBOLS = set(';:,.!?¡¿—…\"«»“”ǃːˈˌˑʼ˞↓↑→↗↘̩')

    output_json = []
    start_ms = 0.0
    for ph_id, dur in zip(ph_ids, durations):
        symbol = id2ph.get(ph_id, f"[UNK_{ph_id}]")
        duration_ms = dur * frame_duration_ms  

        if symbol in SKIPPED_SYMBOLS:
            # Advance time but don't emit a viseme frame
            start_ms += duration_ms
            continue

        viseme_id = phoneme_viseme.get(symbol, 0)
        output_json.append(
            {
                "offset": round(float(start_ms), 3),
                "visemeId": int(viseme_id),
            }
        )
        start_ms += duration_ms

    return output_json
