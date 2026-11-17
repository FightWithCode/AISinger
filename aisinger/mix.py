"""Put the converted vocal back into the song so it sounds produced, not pasted on."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 44100


def load(path: Path, sr: int = SR) -> np.ndarray:
    """Load audio as float32 array shaped (2, n) at ``sr``."""
    from pedalboard.io import AudioFile

    with AudioFile(str(path)).resampled_to(sr) as f:
        audio = f.read(f.frames)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return audio[:2].astype(np.float32)


def save(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    from pedalboard.io import AudioFile

    path.parent.mkdir(parents=True, exist_ok=True)
    with AudioFile(str(path), "w", sr, num_channels=audio.shape[0]) as f:
        f.write(audio)
    return path


def _fit(a: np.ndarray, n: int) -> np.ndarray:
    if a.shape[1] >= n:
        return a[:, :n]
    return np.pad(a, ((0, 0), (0, n - a.shape[1])))


def active_rms(a: np.ndarray, sr: int = SR) -> float:
    """RMS over the louder half of 50 ms frames (ignores silence between lines)."""
    mono = a.mean(axis=0)
    hop = int(sr * 0.05)
    n = len(mono) // hop
    if n == 0:
        return float(np.sqrt(np.mean(mono**2)) + 1e-9)
    frames = mono[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1))
    loud = rms[rms >= np.median(rms)]
    return float(np.sqrt(np.mean(loud**2)) + 1e-9)


def pitch_shift(path: Path, out: Path, semitones: float) -> Path:
    """Transpose a whole track (used to move the song into the new singer's key)."""
    if semitones == 0:
        return path
    from pedalboard import PitchShift

    audio = load(path)
    shifted = PitchShift(semitones=semitones)(audio, SR)
    return save(out, shifted)


def mix(
    vocal: Path,
    instrumental: Path,
    out: Path,
    backing: Path | None = None,
    match_to: Path | None = None,
    vocal_gain_db: float = 0.0,
    backing_gain_db: float = 0.0,
    reverb: float = 0.25,
) -> Path:
    """Mix a (dry) converted vocal over the instrumental.

    ``match_to``: the original lead vocal; the new vocal is level-matched to it
    so the balance of the original mix is preserved.
    ``reverb``: 0 = bone dry, 0.25 = natural room, 0.6 = big hall.
    """
    from pedalboard import Compressor, Gain, HighpassFilter, Limiter, Pedalboard, Reverb

    v = load(vocal)
    inst = load(instrumental)
    n = max(v.shape[1], inst.shape[1])
    v, inst = _fit(v, n), _fit(inst, n)

    if match_to is not None:
        ref = _fit(load(match_to), n)
        v *= active_rms(ref) / active_rms(v)

    chain = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=70),
            Compressor(threshold_db=-20, ratio=2.5, attack_ms=8, release_ms=120),
            Gain(gain_db=vocal_gain_db),
        ]
    )
    if reverb > 0:
        chain.append(
            Reverb(
                room_size=0.15 + 0.6 * reverb,
                damping=0.6,
                wet_level=0.35 * reverb,
                dry_level=1.0,
                width=0.9,
            )
        )
    v = chain(v, SR)

    total = inst + v
    if backing is not None:
        b = _fit(load(backing), n) * (10 ** (backing_gain_db / 20))
        total = total + b

    total = Limiter(threshold_db=-1.0, release_ms=100)(total.astype(np.float32), SR)
    # pedalboard's Limiter has no look-ahead; scale down so nothing clips.
    peak = float(np.abs(total).max())
    if peak > 0.95:
        total *= 0.95 / peak
    return save(out, total)
