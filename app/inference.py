import os, io, math, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import torch
import torchaudio
from joblib import load as joblib_load
try:
    from speechbrain.pretrained import SpeakerRecognition
except Exception:
    # fallback for older layouts
    from speechbrain.inference.speaker import SpeakerRecognition
    from faster_whisper import WhisperModel


# Globals / device

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Keep CPU snappy on Spaces free tier
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
except Exception:
    pass


# Audio utils

VAD = webrtcvad.Vad(2)

def _to_mono_16k(y: np.ndarray, sr: int) -> np.ndarray:
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="kaiser_fast").astype(np.float32)
    return y.astype(np.float32)

def read_uploaded(file) -> np.ndarray:
    """
    Gradio passes a file path for type='filepath' Audio components.
    Accepts either a path-like object with .name or raw bytes.
    """
    if hasattr(file, "name"):
        y, s = sf.read(file.name, dtype="float32", always_2d=False)
    else:
        y, s = sf.read(io.BytesIO(file), dtype="float32", always_2d=False)
    return _to_mono_16k(y, s)

def apply_vad(y: np.ndarray, sr=SR, frame_ms=30) -> np.ndarray:
    frame_len = int(sr * frame_ms / 1000)
    hop = frame_len
    pcm16 = (np.clip(y, -1.0, 1.0) * 32768).astype(np.int16).tobytes()
    out = []
    for i in range(0, len(pcm16), hop * 2):
        frame = pcm16[i:i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        try:
            is_voice = VAD.is_speech(frame, sr)
        except Exception:
            is_voice = True
        if is_voice:
            out.append(np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0)
    return np.concatenate(out) if out else y

def pad_or_trim(y: np.ndarray, seconds: float) -> np.ndarray:
    n = int(SR * seconds)
    return np.pad(y, (0, max(0, n - len(y))))[:n].astype(np.float32)


# Embedding backbone (torchaudio W2V2 XLSR)

BUNDLE = torchaudio.pipelines.WAV2VEC2_XLSR_300M
assert BUNDLE.sample_rate == SR, "Expected 16 kHz backbone"
_w2v = BUNDLE.get_model().to(DEVICE).eval()
for p in _w2v.parameters():
    p.requires_grad = False

def wav2vec2_embed(y: np.ndarray) -> np.ndarray:
    """Return pooled embedding (mean+std of last layer)."""
    wav = torch.from_numpy(y).to(DEVICE).float().unsqueeze(0)  # [1, T]
    with torch.inference_mode():
        feats_list, _ = _w2v.extract_features(wav)             # list of [1, time, feat]
        x = feats_list[-1]
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        feat = torch.cat([mean, std], dim=-1)                  # [1, 2*feat_dim]
    return feat[0].detach().cpu().numpy().astype(np.float32)


# Anti-spoof model (calibrated)

_model_path = Path(__file__).parent / "models" / "antispoof_w2v2lr_robust_calibrated.joblib"
if not _model_path.exists():
    raise FileNotFoundError(
        f"Missing model at '{_model_path}'. "
        "Copy your trained antispoof_w2v2lr_robust_calibrated.joblib to app/models/ (use Git LFS if large)."
    )

state = joblib_load(_model_path)
_scaler = state["scaler"]
_clf = state["clf"]
_cal = state["cal"]

def spoof_probability(y: np.ndarray) -> float:
    """Calibrated P(spoof) in [0,1]."""
    yv = pad_or_trim(apply_vad(y, SR), 4.0)
    emb = wav2vec2_embed(yv)
    z = _scaler.transform([emb])
    return float(_cal.predict_proba(z)[0, 1])


# Speaker verification (SpeechBrain ECAPA)

_verifier = None
def load_speaker_verifier():
    global _verifier
    if _verifier is None:
        _verifier = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(Path(__file__).parent / "pretrained_spk"),
            run_opts={"device": DEVICE},
        )
    return _verifier

def speaker_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return raw similarity score from SpeechBrain verify_files (higher ~ same speaker)."""
    a4 = pad_or_trim(apply_vad(a, SR), 4.0)
    b4 = pad_or_trim(apply_vad(b, SR), 4.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fa, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fb:
        sf.write(fa.name, a4, SR)
        sf.write(fb.name, b4, SR)
        try:
            score, _ = load_speaker_verifier().verify_files(fa.name, fb.name)
        finally:
            for f in (fa.name, fb.name):
                try:
                    os.remove(f)
                except Exception:
                    pass
    return float(score)


# Passphrase ASR (Whisper via faster-whisper)

_whisper = None
def load_whisper(model_size: str = "tiny.en"):
    """
    Use 'tiny.en' by default on CPU Basic to keep latency down.
    You can switch to 'small.en' if you upgrade the hardware.
    """
    global _whisper
    if _whisper is None:
        compute_type = "float16" if DEVICE == "cuda" else "int8"
        _whisper = WhisperModel(model_size, device=DEVICE, compute_type=compute_type)
    return _whisper

def transcribe_digits(y: np.ndarray) -> str:
    """Return only digits recognized from the audio (maps 'oh'/'o'→0, 'for'→4, etc.)."""
    y3 = pad_or_trim(apply_vad(y, SR), 3.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y3, SR)
        segments, _ = load_whisper().transcribe(
            f.name,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 150},
        )
        try:
            os.remove(f.name)
        except Exception:
            pass
    text = " ".join(s.text for s in segments).lower()
    mapping = {
        "zero": "0", "oh": "0", "o": "0",
        "one": "1", "two": "2", "three": "3",
        "four": "4", "for": "4",
        "five": "5", "six": "6", "seven": "7",
        "eight": "8", "ate": "8",
        "nine": "9",
    }
    digits = []
    for tok in text.replace("-", " ").split():
        if tok.isdigit():
            digits.extend(list(tok))
        elif tok in mapping:
            digits.append(mapping[tok])
    return "".join(digits)

# Risk aggregation

@dataclass
class RiskOut:
    speaker_score: float
    spoof_prob: float
    digits_heard: str
    phrase_ok: bool
    risk: float
    label: str

def aggregate_risk(spk_score: float, spoof_prob: float, phrase_ok: bool) -> Tuple[float, str]:
    """
    Aggregate the three checks to an overall risk in [0,1] and a label:
      Safe / Caution / Danger
    """
    # Map raw ECAPA score to [0,1] via logistic squashing.
    spk_sim = 1.0 / (1.0 + math.exp(-spk_score))
    risk = (1.0 - spk_sim) * 0.45 + float(spoof_prob) * 0.45 + (0.0 if phrase_ok else 0.10)
    label = "Safe" if risk < 0.35 else ("Caution" if risk < 0.65 else "Danger")
    return float(risk), label

def run_check(enroll_wav: np.ndarray, probe_wav: np.ndarray, expected_digits: str) -> RiskOut:
    spk = speaker_similarity(enroll_wav, probe_wav)
    spf = spoof_probability(probe_wav)
    heard = transcribe_digits(probe_wav)
    phrase_ok = (heard == expected_digits) if expected_digits else True
    risk, label = aggregate_risk(spk, spf, phrase_ok)
    return RiskOut(spk, spf, heard, phrase_ok, risk, label)
