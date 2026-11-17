"""RVC v2 via Applio (https://github.com/IAHispano/Applio), driven through its CLI.

Trained voices live in ``AISINGER_HOME/voices/<name>/`` (Applio's "logs" dir is
pointed there with ``APPLIO_LOGS_DIR``), so on Colab with Drive mounted every
checkpoint is saved straight to Drive and training can resume after a
disconnect.
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path

from .. import config
from ..utils import AISingerError, log, run
from . import default_gpu, git_checkout, make_venv, torch_index, uv_pip

# Applio's CLI prints these and still exits 0 on failure.
_FAIL = [r"failed for model", r"An error occurred downloading"]


class Applio:
    name = "applio"

    def __init__(self) -> None:
        self.paths = config.get_paths()
        self.dir = self.paths.engines / "applio"
        self.python = self.dir / ".venv" / "bin" / "python"

    # ------------------------------------------------------------------ setup
    @property
    def installed(self) -> bool:
        return (self.dir / ".aisinger-installed").exists() and self.python.exists()

    def install(self, gpu: bool | None = None, force: bool = False) -> None:
        gpu = default_gpu() if gpu is None else gpu
        if self.installed and not force:
            log("Applio already installed.")
            return
        git_checkout(config.APPLIO_REPO, config.APPLIO_REF, self.dir)
        py = make_venv(self.dir, "3.12")
        log("Installing Applio dependencies (a few minutes) ...")
        uv_pip(
            py,
            [
                "-r",
                str(self.dir / "requirements-engine.txt"),
                "--extra-index-url",
                torch_index(gpu, "cu128"),
                "--index-strategy",
                "unsafe-best-match",
            ],
        )
        log("Downloading Applio base models (rmvpe, contentvec, ...) ...")
        # Pretrained G/D weights are fetched lazily for the sample rate you train at.
        self._core(["prerequisites", "--no-pretraineds-hifigan", "--models", "--no-exe"])
        (self.dir / ".aisinger-installed").write_text(config.APPLIO_REF)
        log("Applio ready.")

    def _require(self) -> None:
        if not self.installed:
            raise AISingerError("Applio is not installed. Run:  aisinger setup --engine rvc")

    def _core(self, args: list[str]) -> str:
        env = {
            "APPLIO_LOGS_DIR": str(self.paths.voices),
            "APPLIO_CODE_ROOT": str(self.dir),
            "APPLIO_CONFIG_DIR": str(self.paths.cache / "applio-config"),
            "MPLBACKEND": "Agg",
        }
        return run([self.python, "core.py", *args], cwd=self.dir, env=env, fail_patterns=_FAIL)

    # ---------------------------------------------------------------- voices
    def find_model(self, voice: str) -> tuple[Path, Path | None]:
        """Return (pth, index) for a voice name or a direct .pth path."""
        p = Path(voice).expanduser()
        if p.suffix == ".pth" and p.exists():
            idx = next(iter(sorted(p.parent.glob("*.index"))), None)
            return p, idx
        vdir = self.paths.voice_dir(voice)
        if not vdir.exists():
            raise AISingerError(
                f"No voice named '{voice}' in {self.paths.voices}. "
                "Train one (aisinger train) or import one (aisinger import)."
            )
        candidates = [
            f for f in vdir.glob("*.pth") if not re.match(r"^[GD]_", f.name)
        ]
        if not candidates:
            raise AISingerError(f"Voice '{voice}' has no trained model (.pth) yet.")

        def epoch_of(f: Path) -> int:
            m = re.search(r"_(\d+)e_(\d+)s\.pth$", f.name)
            return int(m.group(1)) if m else 10**9  # imported/renamed models win

        pinned = vdir / "use.txt"  # optional: name of the .pth to use
        if pinned.exists():
            chosen = vdir / pinned.read_text().strip()
            if not chosen.exists():
                raise AISingerError(f"{pinned} points to missing file {chosen.name}")
        else:
            chosen = max(candidates, key=lambda f: (epoch_of(f), f.stat().st_mtime))
        indexes = sorted(vdir.glob("*.index"), key=lambda f: ("added" not in f.name, f.name))
        return chosen, (indexes[0] if indexes else None)

    def import_model(self, name: str, source: str) -> Path:
        """Import an existing RVC model: .pth (+ .index), a folder, a .zip, or a URL."""
        vdir = self.paths.voice_dir(name)
        vdir.mkdir(parents=True, exist_ok=True)
        if re.match(r"^https?://", source):
            tmp = self.paths.cache / "downloads" / name
            tmp.mkdir(parents=True, exist_ok=True)
            self._require()
            before = set(self.paths.voices.glob("*"))
            self._core(["download", "--model-link", source])
            new_dirs = [d for d in set(self.paths.voices.glob("*")) - before if d.is_dir()]
            if new_dirs and new_dirs[0] != vdir:
                for f in new_dirs[0].iterdir():
                    shutil.move(str(f), vdir / f.name)
                shutil.rmtree(new_dirs[0], ignore_errors=True)
        else:
            src = Path(source).expanduser()
            if not src.exists():
                raise AISingerError(f"Not found: {src}")
            files: list[Path]
            if src.suffix == ".zip":
                with zipfile.ZipFile(src) as z:
                    z.extractall(vdir / "_unzipped")
                files = list((vdir / "_unzipped").rglob("*"))
            elif src.is_dir():
                files = list(src.rglob("*"))
            else:
                files = [src] + list(src.parent.glob("*.index"))
            for f in files:
                if f.suffix in (".pth", ".index") and not re.match(r"^[GD]_", f.name):
                    shutil.copy2(f, vdir / f.name)
            shutil.rmtree(vdir / "_unzipped", ignore_errors=True)
        pth, idx = self.find_model(name)
        log(f"Imported voice '{name}': {pth.name} (index: {idx.name if idx else 'none'})")
        return vdir

    # -------------------------------------------------------------- training
    def train(
        self,
        name: str,
        dataset: Path,
        epochs: int = 300,
        save_every: int = 25,
        batch_size: int = 8,
        sample_rate: int = 40000,
        f0_method: str = "rmvpe",
        vocoder: str = "HiFi-GAN",
        fresh: bool = False,
    ) -> Path:
        self._require()
        vdir = self.paths.voice_dir(name)
        if fresh and vdir.exists():
            log(f"--fresh: deleting previous training files in {vdir}")
            shutil.rmtree(vdir)
        gpu = "0" if default_gpu() else "-"
        cores = str(max(1, min(os.cpu_count() or 1, 16)))
        resuming = any(vdir.glob("G_*.pth"))
        if resuming:
            log("Found checkpoints, resuming training (use --fresh to start over).")
        else:
            log("Step 1/3: slicing dataset ...")
            self._core(
                [
                    "preprocess",
                    "--model-name", name,
                    "--dataset-path", str(dataset),
                    "--sample-rate", str(sample_rate),
                    "--cpu-cores", cores,
                    "--cut-preprocess", "Automatic",
                    "--normalization-mode", "post",
                ]
            )
            log("Step 2/3: extracting pitch + content features ...")
            self._core(
                [
                    "extract",
                    "--model-name", name,
                    "--f0-method", f0_method,
                    "--cpu-cores", cores,
                    "--gpu", gpu,
                    "--sample-rate", str(sample_rate),
                    "--embedder-model", "contentvec",
                ]
            )
        log(f"Step 3/3: training for {epochs} epochs ...")
        args = [
            "train",
            "--model-name", name,
            "--vocoder", vocoder,
            "--save-every-epoch", str(save_every),
            "--save-only-latest",  # keep only the newest big G/D checkpoint (saves Drive space)
            "--total-epoch", str(epochs),
            "--sample-rate", str(sample_rate),
            "--batch-size", str(batch_size),
            "--gpu", "0" if gpu == "0" else "-",
            "--index-algorithm", "Auto",
        ]
        self._core(args)
        pth, idx = self.find_model(name)
        if idx is None:
            self._core(["index", "--model-name", name, "--index-algorithm", "Auto"])
        log(f"Voice '{name}' trained: {pth}")
        return pth

    # ------------------------------------------------------------- inference
    def convert(
        self,
        input_path: Path,
        output_path: Path,
        voice: str,
        pitch: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.33,
        f0_method: str = "rmvpe",
        volume_envelope: float = 1.0,
    ) -> Path:
        self._require()
        pth, idx = self.find_model(voice)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._core(
            [
                "infer",
                "--input-path", str(input_path),
                "--output-path", str(output_path),
                "--pth-path", str(pth),
                "--index-path", str(idx) if idx else "",
                "--pitch", str(pitch),
                "--index-rate", str(index_rate if idx else 0),
                "--protect", str(protect),
                "--f0-method", f0_method,
                "--volume-envelope", str(volume_envelope),
                "--export-format", "WAV",
            ]
        )
        if not output_path.exists():
            raise AISingerError(f"RVC conversion did not produce {output_path}")
        return output_path

    def speak(
        self,
        text: str,
        voice: str,
        output_path: Path,
        tts_voice: str = "hi-IN-MadhurNeural",
        tts_rate: int = 0,
        pitch: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.33,
    ) -> Path:
        """Text -> speech (edge-tts) -> converted into the cloned voice."""
        self._require()
        pth, idx = self.find_model(voice)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        text_file = output_path.with_suffix(".txt")
        text_file.write_text(text, encoding="utf-8")
        self._core(
            [
                "tts",
                "--tts-file", str(text_file),
                "--tts-text", text,
                "--tts-voice", tts_voice,
                "--tts-rate", str(tts_rate),
                "--output-tts-path", str(output_path.with_name(output_path.stem + "_tts.wav")),
                "--output-rvc-path", str(output_path),
                "--pth-path", str(pth),
                "--index-path", str(idx) if idx else "",
                "--pitch", str(pitch),
                "--index-rate", str(index_rate if idx else 0),
                "--protect", str(protect),
                "--export-format", "WAV",
            ]
        )
        if not output_path.exists():
            raise AISingerError(f"TTS did not produce {output_path}")
        return output_path
