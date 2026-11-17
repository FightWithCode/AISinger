"""Small shared helpers: running commands, audio file handling."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".webm", ".mp4"}


class AISingerError(RuntimeError):
    """A user-facing error with a readable message."""


def log(msg: str) -> None:
    print(f"[aisinger] {msg}", flush=True)


def run(
    cmd: Sequence[str | os.PathLike],
    cwd: str | os.PathLike | None = None,
    env: dict | None = None,
    fail_patterns: Iterable[str] = (),
) -> str:
    """Run a command, streaming its output live, and return the output.

    ``fail_patterns`` catches tools that print an error but still exit 0
    (Applio's CLI does this).
    """
    cmd = [str(c) for c in cmd]
    full_env = {**os.environ, **(env or {})}
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
    code = proc.wait()
    out = "".join(lines)
    if code != 0:
        raise AISingerError(f"Command failed (exit {code}): {' '.join(cmd[:4])} ...")
    for pat in fail_patterns:
        if re.search(pat, out, flags=re.IGNORECASE):
            raise AISingerError(f"Command reported a failure ({pat!r}): {' '.join(cmd[:4])} ...")
    return out


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\-]+", "_", text, flags=re.UNICODE).strip("_")
    return text[:80] or "song"


def collect_audio(inputs: Iterable[str | os.PathLike]) -> list[Path]:
    """Expand files / folders into a sorted list of audio files."""
    files: list[Path] = []
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            raise AISingerError(f"Not found: {p}")
    if not files:
        raise AISingerError("No audio files found in the given inputs.")
    return files


def to_wav(src: Path, dst: Path, sr: int | None = None, mono: bool = False) -> Path:
    """Decode any audio/video file to WAV using ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise AISingerError("ffmpeg is required (apt install ffmpeg / brew install ffmpeg).")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn"]
    if sr:
        cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)
    return dst


def to_mp3(src: Path, dst: Path, bitrate: str = "320k") -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", bitrate, str(dst)], check=True
    )
    return dst


def duration_seconds(path: Path) -> float:
    import soundfile as sf

    try:
        info = sf.info(str(path))
        return info.frames / info.samplerate
    except Exception:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return float(out or 0)
