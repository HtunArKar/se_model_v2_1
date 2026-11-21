# -*- coding: utf-8 -*-
"""
Replicate model: Thai speech enhancement (SE model v2_1).

    input : audio    noisy speech file (WAV, FLAC, OGG, MP3); converted to 16 kHz mono if needed
            mix_back fraction of the noisy input mixed back into the output (0 = none)
    output: enhanced speech, WAV, 16 kHz mono, 32-bit float

Same model code as services/se_service (se_core.py, copied here by prepare.py), so the result is the
same as the SE service on the same kind of GPU.
"""

import os
import sys
import tempfile
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parent
os.environ.setdefault("DEMO_ROOT", str(ROOT))           # se_core finds SE_Model, checkpoint, config here
sys.path.insert(0, str(ROOT / "services"))
sys.path.insert(0, str(ROOT / "services" / "se_service"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from cog import BasePredictor, Input, Path  # noqa: E402

SAMPLE_RATE = 16000
MAX_SECONDS = 600.0


def load_audio(path, sr=SAMPLE_RATE):
    """Any file soundfile can read -> mono float32 at 16 kHz (same as the gateway and app_1.py)."""
    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = data[:, 0].copy() if data.shape[1] == 1 else data.mean(axis=1, dtype=np.float32)
    if file_sr != sr:
        import torch
        import torchaudio
        audio = torchaudio.functional.resample(torch.from_numpy(audio), orig_freq=file_sr, new_freq=sr).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


class Predictor(BasePredictor):

    def setup(self):
        from se_core import SpeechEnhancer
        self.se = SpeechEnhancer("v2_1")                # GPU if the machine has one, otherwise CPU
        print(f"SE model v2_1 loaded on {self.se.device_name}")

    def predict(
        self,
        audio: Path = Input(description="Noisy speech: WAV, FLAC, OGG or MP3. Converted to 16 kHz mono if needed."),
        mix_back: float = Input(description="Fraction of the noisy input mixed back into the output (0 = none)",
                                default=0.0, ge=0.0, le=1.0),
    ) -> Path:
        x = load_audio(audio)
        seconds = len(x) / SAMPLE_RATE
        if seconds < 0.1:
            raise ValueError("the audio is shorter than 0.1 s")
        if seconds > MAX_SECONDS:
            raise ValueError(f"the audio is longer than {MAX_SECONDS:.0f} s")

        enhanced, compute = self.se.process(x, mix_back)
        print(f"{seconds:.2f} s of audio enhanced in {compute:.3f} s on {self.se.device_name}")

        out = Path(tempfile.mkdtemp()) / "enhanced.wav"
        sf.write(str(out), enhanced, SAMPLE_RATE, subtype="FLOAT")
        return out
