"""Stem separation with python-audio-separator (UVR / RoFormer models).

Steps used by the cover pipeline:

1. song      -> vocals + instrumental        (BS-RoFormer)
2. vocals    -> lead + backing               (Mel-RoFormer karaoke)   [optional]
3. lead      -> dry lead (+ reverb removed)  (UVR DeEcho-DeReverb)     [optional]
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from . import config
from .utils import AISingerError, log

_separator = None


def _get_separator():
    global _separator
    if _separator is None:
        try:
            from audio_separator.separator import Separator
        except ImportError as e:  # pragma: no cover
            raise AISingerError(f"audio-separator is not installed correctly: {e}") from e
        paths = config.get_paths()
        _separator = Separator(
            model_file_dir=str(paths.separator_models),
            output_format="WAV",
            # RoFormer models are 44.1 kHz; keep everything at that rate.
            sample_rate=44100,
        )
    return _separator


def _stem_name(filename: str) -> str:
    m = re.search(r"_\(([^)]+)\)", Path(filename).name)
    return (m.group(1) if m else Path(filename).stem).strip().lower()


def run_model(input_path: Path, model: str, out_dir: Path) -> dict[str, Path]:
    """Run one separation model; return {stem_name_lowercase: path}."""
    sep = _get_separator()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        sep.output_dir = tmp
        if getattr(sep, "model_instance", None) is not None:
            sep.model_instance.output_dir = tmp
        log(f"Separating {input_path.name} with {model} ...")
        sep.load_model(model_filename=model)
        produced = sep.separate(str(input_path))
        stems: dict[str, Path] = {}
        for f in produced:
            fp = Path(f)
            if not fp.is_absolute():
                fp = Path(tmp) / fp
            stems[_stem_name(fp.name)] = fp
        if not stems:
            raise AISingerError(f"Separation produced no files for {input_path}")
        # Move out of the temp dir with stable names.
        final: dict[str, Path] = {}
        for stem, fp in stems.items():
            dst = out_dir / f"{re.sub(r'[^a-z0-9]+', '_', stem).strip('_')}.wav"
            shutil.move(str(fp), dst)
            final[stem] = dst
    return final


def _pick(stems: dict[str, Path], *wanted: str, exclude: tuple[str, ...] = ()) -> Path:
    for w in wanted:
        for name, p in stems.items():
            if w in name and not any(x in name for x in exclude):
                return p
    raise AISingerError(f"Could not find stem {wanted} in {list(stems)}")


def split_vocals(song: Path, out_dir: Path, model: str = config.SEP_VOCALS_MODEL) -> tuple[Path, Path]:
    """Return (vocals, instrumental)."""
    vocals, inst = out_dir / "vocals.wav", out_dir / "instrumental.wav"
    if vocals.exists() and inst.exists():
        log("Using cached vocals/instrumental split.")
        return vocals, inst
    stems = run_model(song, model, out_dir / "_raw_split")
    _pick(stems, "vocal").replace(vocals)
    _pick(stems, "instrumental", "other", "no vocal", exclude=("no instrumental",)).replace(inst)
    shutil.rmtree(out_dir / "_raw_split", ignore_errors=True)
    return vocals, inst


def split_backing(vocals: Path, out_dir: Path, model: str = config.SEP_KARAOKE_MODEL) -> tuple[Path, Path]:
    """Return (lead_vocals, backing_vocals)."""
    lead, backing = out_dir / "lead.wav", out_dir / "backing.wav"
    if lead.exists() and backing.exists():
        log("Using cached lead/backing split.")
        return lead, backing
    stems = run_model(vocals, model, out_dir / "_raw_karaoke")
    _pick(stems, "vocal", "lead", "karaoke", exclude=("no vocal", "instrumental")).replace(lead)
    _pick(stems, "instrumental", "backing", "other", "no vocal").replace(backing)
    shutil.rmtree(out_dir / "_raw_karaoke", ignore_errors=True)
    return lead, backing


def dereverb(vocals: Path, out_path: Path, model: str = config.SEP_DEREVERB_MODEL) -> Path:
    """Remove echo/reverb from a vocal track; returns the dry vocal."""
    if out_path.exists():
        log(f"Using cached {out_path.name}.")
        return out_path
    raw = out_path.parent / f"_raw_dereverb_{out_path.stem}"
    stems = run_model(vocals, model, raw)
    _pick(stems, "no reverb", "noreverb", "dry", "no echo", exclude=("no dry",)).replace(out_path)
    shutil.rmtree(raw, ignore_errors=True)
    return out_path


def denoise(vocals: Path, out_path: Path, model: str = config.SEP_DENOISE_MODEL) -> Path:
    if out_path.exists():
        return out_path
    raw = out_path.parent / f"_raw_denoise_{out_path.stem}"
    stems = run_model(vocals, model, raw)
    _pick(stems, "dry", "no noise", exclude=("no dry",)).replace(out_path)
    shutil.rmtree(raw, ignore_errors=True)
    return out_path
