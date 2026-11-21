# -*- coding: utf-8 -*-
"""
The SE model itself (no web code): noisy 16 kHz speech -> enhanced 16 kHz speech.

Same steps as app_1.py, which copies them from evaluate.py / evaluate_chunk.py:
STFT -> harmonic mask (HRPS) -> power compression -> SE_Model -> iSTFT, on 3 s chunks with
0.5 s overlap joined by a linear cross-fade (perform_denoising_overlap() of evaluate_chunk.py).
"""

import importlib
import json
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.contract import DEMO_ROOT, SAMPLE_RATE

sys.path.insert(0, str(DEMO_ROOT))
from Harmonic_Mask.harmonic_residual_mask import hrps  # noqa: E402

# name : (python module that defines SE_Model, checkpoint file)
SE_MODELS = {
    "v2_1": ("SE_Model.se_model_v2_1", DEMO_ROOT / "Checkpoints_v2_1_lossTest" / "epoch_50.pth"),
    "tftf": ("SE_Model.se_model_v2_1_tftf", DEMO_ROOT / "Checkpoints_T_F_T_F" / "epoch_50.pth"),
}
DEFAULT_SE_MODEL = "v2_1"
CONFIG_PATH = DEMO_ROOT / "Configuration" / "config_1.json"

SE_CHUNK_SECONDS = 3.0
SE_OVERLAP_SECONDS = 0.5


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def power_compress(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    mag = mag ** 0.3
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], 1)


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    return torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], -1)


class MemorySafeBlock(nn.Module):
    """Runs a Conformer block on a few sequences at a time (same output, lower peak memory). See app_1.py."""

    def __init__(self, block, heads, budget_bytes):
        super().__init__()
        self.block = block
        self.heads = heads
        self.budget_bytes = budget_bytes

    def forward(self, x, **kwargs):
        n_seq, n, _ = x.shape
        per_seq = self.heads * n * n * 4 * 6
        step = max(1, int(self.budget_bytes // per_seq))
        if step >= n_seq:
            return self.block(x, **kwargs)
        return torch.cat([self.block(x[i:i + step], **kwargs) for i in range(0, n_seq, step)], dim=0)


class SpeechEnhancer:

    def __init__(self, name=DEFAULT_SE_MODEL, device=None):
        if name not in SE_MODELS:
            raise ValueError(f"unknown SE model '{name}', choose from {list(SE_MODELS)}")
        self.name = name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        self.lock = threading.Lock()            # one request at a time on the GPU

        with open(CONFIG_PATH) as f:
            self.cfg = AttrDict(json.load(f))
        self.sr = int(self.cfg.sampling_rate)
        if self.sr != SAMPLE_RATE:
            raise ValueError(f"the SE config says {self.sr} Hz, the service contract is {SAMPLE_RATE} Hz")

        self._load()
        self.enhance(0.01 * torch.randn(self.sr, generator=torch.Generator().manual_seed(0)).numpy())  # warm-up
        self.free()

    def _load(self):
        h = self.cfg
        module_name, ckpt_path = SE_MODELS[self.name]
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"SE checkpoint not found: {ckpt_path}")
        self.checkpoint = ckpt_path

        SE_Model = importlib.import_module(module_name).SE_Model
        model = SE_Model(in_channels=2, channels=h.channels, F_bins=int(h.n_fft / 2 + 1), r=h.r, G=h.G,
                         dilation_enc=h.dilation_enc, dilation_dec=h.dilation_dec,
                         down_factor=h.down_factor, mask_cond=True, resample="shuffle",
                         conf_num_layers=h.conf_num_layers, conv_kernel_size=h.conv_kernel_size,
                         heads=h.head, ff_mult=h.ff_mult, expansion_factor=h.expansion_factor,
                         attn_dropout=h.attn_dropout, ff_dropout=h.ff_dropout, conv_dropout=h.conv_dropout)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        budget = (1.0 if self.device.type == "cuda" else 2.0) * 1024 ** 3
        for i in range(len(model.con_blocks)):
            model.con_blocks[i] = MemorySafeBlock(model.con_blocks[i], heads=h.head, budget_bytes=budget)

        self.model = model.to(self.device)
        self.window = torch.hann_window(h.n_fft, device=self.device)
        self.params = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def _enhance_once(self, wav):
        """One pass through the SE model (perform_denoising() in evaluate_chunk.py)."""
        h = self.cfg
        noisy = wav.unsqueeze(0).to(self.device)
        length = noisy.size(-1)

        padding = int(np.ceil(length / h.hop)) * h.hop - length
        if padding > 0:
            noisy = F.pad(noisy, (0, padding))

        noisy_spec = torch.stft(noisy, h.n_fft, h.hop, win_length=h.n_fft, window=self.window,
                                center=True, return_complex=True)
        _, Freq, T = noisy_spec.shape

        noisy_harmonic_mask, _ = hrps(noisy_spec, Fs=h.sampling_rate, N=h.n_fft, H=h.hop,
                                      L_h=h.L_h_sec, L_p=h.L_p_Hz, beta=h.beta)

        inputs = power_compress(noisy_spec).permute(0, 1, 3, 2)
        harmonic_mask = noisy_harmonic_mask.permute(0, 2, 1).unsqueeze(1)

        est_spec = self.model(inputs, harmonic_mask)[:, :, :T, :Freq]

        est_real = est_spec[:, 0:1, :, :].permute(0, 1, 3, 2)
        est_imag = est_spec[:, 1:2, :, :].permute(0, 1, 3, 2)

        spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        if not torch.is_complex(spec_uncompress):
            spec_uncompress = torch.view_as_complex(spec_uncompress)

        est_wav = torch.istft(spec_uncompress, h.n_fft, h.hop, win_length=h.n_fft, window=self.window,
                              center=True)
        return torch.flatten(est_wav)[:length].cpu()

    def enhance(self, audio):
        """audio: 1-D float32 numpy array at 16 kHz -> enhanced audio, same length (perform_denoising_overlap())."""
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        n = wav.numel()
        chunk = int(SE_CHUNK_SECONDS * self.sr)
        ov = int(SE_OVERLAP_SECONDS * self.sr)

        def run(piece):
            out = self._enhance_once(piece)
            if out.numel() < piece.numel():
                out = torch.cat([out, piece[out.numel():]])
            return out[:piece.numel()]

        if n <= chunk:
            return run(wav).numpy()

        fade_in = torch.linspace(0.0, 1.0, ov)
        fade_out = 1.0 - fade_in
        out = torch.zeros(n)
        weight = torch.zeros(n)

        for start in range(0, n, chunk - ov):
            end = min(start + chunk, n)
            if end == n and end - start < chunk:
                start = max(0, n - chunk)

            piece = run(wav[start:end])

            w = torch.ones(end - start)
            k = min(ov, end - start)
            if start > 0:
                w[:k] = fade_in[:k]
            if end < n:
                w[-k:] = fade_out[-k:]

            out[start:end] += piece * w
            weight[start:end] += w

        return (out / weight.clamp_min(1e-8)).numpy()

    def process(self, audio, mix_back=0.0):
        """The service call: enhance, then optionally mix some of the noisy input back (evaluate.py formula)."""
        with self.lock:
            t0 = time.perf_counter()
            try:
                enhanced = self.enhance(audio)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
            finally:
                self.free()
            if mix_back > 0:
                enhanced = (1.0 - mix_back) * enhanced + mix_back * audio[:len(enhanced)]
            return enhanced.astype(np.float32), time.perf_counter() - t0

    def free(self):
        """Give cached GPU memory back after each request: the ASR and LLM services may share this GPU."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
