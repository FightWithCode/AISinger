"""Paths and settings.

Two roots:

* ``AISINGER_HOME`` – your data: voices (trained models), datasets, outputs.
  Put this on Google Drive in Colab so trained voices survive disconnects.
* ``AISINGER_CACHE`` – re-downloadable stuff: engine code + their virtualenvs,
  separation models. Keep this on fast local disk.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

# Pinned engine versions. Bump deliberately; the wrappers depend on their CLIs.
APPLIO_REPO = "https://github.com/IAHispano/Applio.git"
APPLIO_REF = "3.7.1"
SEEDVC_REPO = "https://github.com/Plachtaa/seed-vc.git"
SEEDVC_REF = "51383efd921027683c89e5348211d93ff12ac2a8"

# Separation models (names as listed by `audio-separator --list_models`).
SEP_VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # vocals / instrumental
SEP_KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"  # lead / backing
SEP_DEREVERB_MODEL = "UVR-DeEcho-DeReverb.pth"  # removes echo + reverb (great for old songs)
SEP_DENOISE_MODEL = "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"


def _in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


@dataclass
class Paths:
    home: Path
    cache: Path

    @property
    def voices(self) -> Path:
        return self.home / "voices"

    @property
    def datasets(self) -> Path:
        return self.home / "datasets"

    @property
    def outputs(self) -> Path:
        return self.home / "outputs"

    @property
    def engines(self) -> Path:
        return self.cache / "engines"

    @property
    def separator_models(self) -> Path:
        return self.cache / "separator-models"

    def voice_dir(self, name: str) -> Path:
        return self.voices / name

    def ensure(self) -> "Paths":
        for p in (self.voices, self.datasets, self.outputs, self.engines, self.separator_models):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    in_colab = _in_colab()
    default_home = Path("/content/AISinger") if in_colab else Path.home() / "AISinger"
    default_cache = Path("/content/aisinger-cache") if in_colab else Path.home() / ".cache" / "aisinger"
    home = Path(os.environ.get("AISINGER_HOME", default_home)).expanduser()
    cache = Path(os.environ.get("AISINGER_CACHE", default_cache)).expanduser()
    return Paths(home=home, cache=cache).ensure()


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None and os.system("nvidia-smi -L >/dev/null 2>&1") == 0
