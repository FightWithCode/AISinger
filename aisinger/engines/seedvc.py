"""Seed-VC (https://github.com/Plachtaa/seed-vc): zero-shot singing voice conversion.

Zero-shot = no training. Give it 5–25 s of clean (dry, music-free) audio of the
target voice and it converts any vocal into that voice. Optional fine-tuning on
a few minutes of data improves similarity further.

A Seed-VC voice is a folder ``AISINGER_HOME/voices/<name>/`` containing
``reference.wav`` and, if fine-tuned, ``seedvc_run/ft_model.pth``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from .. import config
from ..utils import AISingerError, collect_audio, duration_seconds, log, run
from . import default_gpu, git_checkout, make_venv, torch_index, uv_pip

SINGING_CONFIG = "configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"

# Curated from seed-vc/requirements.txt: inference + training only (no GUIs / eval tools).
_DEPS = [
    "torch==2.4.0",
    "torchaudio==2.4.0",
    "scipy==1.13.1",
    "librosa==0.10.2",
    "huggingface-hub>=0.28.1,<1.0",
    "munch==4.0.0",
    "einops==0.8.0",
    "descript-audio-codec==1.0.0",
    "pydub==0.25.1",
    "transformers==4.46.3",
    "soundfile==0.12.1",
    "numpy==1.26.4",
    "hydra-core==1.3.2",
    "pyyaml",
    "python-dotenv",
    "accelerate<1.5",
    "matplotlib",
]

MAX_REF_SECONDS = 25


class SeedVC:
    name = "seedvc"

    def __init__(self) -> None:
        self.paths = config.get_paths()
        self.dir = self.paths.engines / "seedvc"
        self.python = self.dir / ".venv" / "bin" / "python"

    @property
    def installed(self) -> bool:
        return (self.dir / ".aisinger-installed").exists() and self.python.exists()

    def install(self, gpu: bool | None = None, force: bool = False) -> None:
        gpu = default_gpu() if gpu is None else gpu
        if self.installed and not force:
            log("Seed-VC already installed.")
            return
        git_checkout(config.SEEDVC_REPO, config.SEEDVC_REF, self.dir)
        py = make_venv(self.dir, "3.10")
        log("Installing Seed-VC dependencies (a few minutes) ...")
        uv_pip(
            py,
            [*_DEPS, "--extra-index-url", torch_index(gpu, "cu121"), "--index-strategy", "unsafe-best-match"],
        )
        (self.dir / ".aisinger-installed").write_text(config.SEEDVC_REF)
        log("Seed-VC ready (model weights download on first use).")

    def _require(self) -> None:
        if not self.installed:
            raise AISingerError("Seed-VC is not installed. Run:  aisinger setup --engine seedvc")

    # ------------------------------------------------------------ references
    def set_reference(self, name: str, clip: Path, start: float = 0.0) -> Path:
        """Store a reference clip (trimmed to 25 s, mono 44.1 kHz) for a voice."""
        vdir = self.paths.voice_dir(name)
        vdir.mkdir(parents=True, exist_ok=True)
        ref = vdir / "reference.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-ss", str(start), "-i", str(clip),
                "-t", str(MAX_REF_SECONDS), "-ac", "1", "-ar", "44100", str(ref),
            ],
            check=True,
        )
        log(f"Reference for '{name}' saved ({duration_seconds(ref):.1f}s): {ref}")
        return ref

    def _resolve(self, voice: str) -> tuple[Path, Path | None, Path | None]:
        """Return (reference, checkpoint, config) for a voice name or a clip path."""
        p = Path(voice).expanduser()
        if p.is_file():
            return p, None, None
        vdir = self.paths.voice_dir(voice)
        ref = vdir / "reference.wav"
        if not ref.exists():
            raise AISingerError(
                f"Voice '{voice}' has no reference clip. Add one with:\n"
                f"  aisinger reference {voice} path/to/clean_clip.wav"
            )
        ckpt = vdir / "seedvc_run" / "ft_model.pth"
        cfg = vdir / "seedvc_run" / Path(SINGING_CONFIG).name
        if ckpt.exists() and cfg.exists():
            return ref, ckpt, cfg
        return ref, None, None

    # ------------------------------------------------------------- inference
    def convert(
        self,
        input_path: Path,
        output_path: Path,
        voice: str,
        pitch: int = 0,
        diffusion_steps: int = 40,
        cfg_rate: float = 0.7,
    ) -> Path:
        self._require()
        ref, ckpt, cfg = self._resolve(voice)
        out_dir = output_path.parent / "_seedvc_tmp"
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        args = [
            self.python, "inference.py",
            "--source", str(input_path.resolve()),
            "--target", str(ref.resolve()),
            "--output", str(out_dir.resolve()),
            "--diffusion-steps", str(diffusion_steps),
            "--length-adjust", "1.0",
            "--inference-cfg-rate", str(cfg_rate),
            "--f0-condition", "True",  # singing mode: follow the source melody
            "--auto-f0-adjust", "False",
            "--semi-tone-shift", str(pitch),
            "--fp16", "True" if default_gpu() else "False",
        ]
        if ckpt and cfg:
            args += ["--checkpoint", str(ckpt), "--config", str(cfg)]
        run(args, cwd=self.dir)
        produced = sorted(out_dir.glob("*.wav"), key=lambda f: f.stat().st_mtime)
        if not produced:
            raise AISingerError("Seed-VC produced no output.")
        shutil.move(str(produced[-1]), output_path)
        shutil.rmtree(out_dir, ignore_errors=True)
        return output_path

    # -------------------------------------------------------------- training
    def train(self, name: str, dataset: Path, steps: int = 1000, batch_size: int = 2, fresh: bool = False) -> Path:
        """Fine-tune Seed-VC's singing model on a voice dataset (a few minutes of audio)."""
        self._require()
        vdir = self.paths.voice_dir(name)
        run_dir = vdir / "seedvc_run"
        if fresh:
            shutil.rmtree(run_dir, ignore_errors=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Seed-VC writes to ./runs/<name>; point that at Drive-backed storage.
        link = self.dir / "runs" / name
        link.parent.mkdir(exist_ok=True)
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(run_dir.resolve(), target_is_directory=True)

        if not (vdir / "reference.wav").exists():
            longest = max(collect_audio([dataset]), key=duration_seconds)
            self.set_reference(name, longest)

        run(
            [
                self.python, "train.py",
                "--config", SINGING_CONFIG,
                "--dataset-dir", str(dataset.resolve()),
                "--run-name", name,
                "--batch-size", str(batch_size),
                "--max-steps", str(steps),
                "--max-epochs", "1000",
                "--save-every", str(max(100, steps // 4)),
                "--num-workers", "0",
                "--gpu", "0",
            ],
            cwd=self.dir,
            env={"HF_HUB_CACHE": str(self.dir / "checkpoints" / "hf_cache")},
        )
        ckpt = run_dir / "ft_model.pth"
        if not ckpt.exists():
            raise AISingerError(f"Seed-VC training finished but {ckpt} is missing.")
        log(f"Seed-VC voice '{name}' fine-tuned: {ckpt}")
        return ckpt
