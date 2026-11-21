# -*- coding: utf-8 -*-
"""
The audio contract shared by every service and the gateway.

Audio between the services is always a WAV file: 16 kHz, mono, 32-bit float samples.
Float samples carry the SE model's output exactly (16-bit samples would round it).
Only the gateway reads user files and resamples them; the services refuse anything else,
so no service repeats the gateway's preprocessing.
"""

import io
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
WAV_MEDIA_TYPE = "audio/wav"

SERVICES_DIR = Path(__file__).resolve().parents[1]
# The folder that holds SE_Model, Harmonic_Mask, Configuration, the checkpoints and T_Whisper
DEMO_ROOT = Path(os.environ.get("DEMO_ROOT", SERVICES_DIR.parent)).resolve()

# For the API documentation page (/docs) of the services: the request body is a WAV file
WAV_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {WAV_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}},
    }
}


class AudioContractError(ValueError):
    """The request body is not 16 kHz mono WAV audio."""


def wav_bytes(audio, sr=SAMPLE_RATE):
    """1-D float array -> WAV file bytes (32-bit float samples)."""
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, dtype=np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def read_wav_bytes(data, sr=SAMPLE_RATE):
    """WAV file bytes -> 1-D float32 numpy array. Refuses other sample rates and multi-channel audio."""
    if not data:
        raise AudioContractError("the request body is empty: send a WAV file")
    try:
        audio, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioContractError(f"the request body is not a readable WAV file ({e})")
    if file_sr != sr:
        raise AudioContractError(f"expected {sr} Hz audio, got {file_sr} Hz (resampling is the gateway's job)")
    if audio.shape[1] != 1:
        raise AudioContractError(f"expected mono audio, got {audio.shape[1]} channels")
    if audio.shape[0] == 0:
        raise AudioContractError("the audio has no samples")
    return np.ascontiguousarray(audio[:, 0])


def utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def keep_full_speed():
    """
    Windows 11 slows a program down ("efficiency mode") while its window is minimised or behind
    other windows. Measured on the RTX 4050 laptop: recognition 8.5 s instead of 1.6 s, with the
    GPU held at low power. This asks Windows not to do that for this program. Does nothing on Linux.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        class PowerThrottling(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                   wintypes.DWORD]
        state = PowerThrottling(1, 0x1, 0)          # control the execution speed; 0 = never throttle
        kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4,      # 4 = ProcessPowerThrottling
                                       ctypes.byref(state), ctypes.sizeof(state))
    except Exception:
        pass                                        # older Windows: still works, only slower
