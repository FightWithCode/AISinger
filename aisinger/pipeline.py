"""High-level workflows: dataset preparation, cover generation, vocal conversion."""

from __future__ import annotations

import shutil
from pathlib import Path

from . import config, mix, separate
from .utils import AISingerError, collect_audio, duration_seconds, log, slugify, to_mp3, to_wav


def get_engine(name: str):
    name = name.lower()
    if name in ("rvc", "applio"):
        from .engines.applio import Applio

        return Applio()
    if name in ("seedvc", "seed-vc", "zero-shot", "zeroshot"):
        from .engines.seedvc import SeedVC

        return SeedVC()
    raise AISingerError(f"Unknown engine '{name}' (use 'rvc' or 'seedvc').")


# --------------------------------------------------------------------- dataset
def prepare_dataset(
    name: str,
    inputs: list[str],
    isolate: bool = False,
    lead_only: bool = False,
    dereverb: bool = False,
    denoise: bool = False,
) -> Path:
    """Build ``datasets/<name>/`` from raw recordings or songs.

    * Your own a-cappella recordings: no flags needed (``--dereverb`` if the room echoes).
    * Full songs of a singer (e.g. Kishore Kumar): ``--isolate --lead-only --dereverb``.
    """
    paths = config.get_paths()
    out = paths.datasets / name
    out.mkdir(parents=True, exist_ok=True)
    work = paths.cache / "dataset-work" / name
    files = collect_audio(inputs)
    log(f"Preparing {len(files)} file(s) for voice '{name}' ...")
    for i, f in enumerate(files, 1):
        stem = f"{i:03d}_{slugify(f.stem)}"
        target = out / f"{stem}.wav"
        if target.exists():
            continue
        fw = work / stem
        cur = to_wav(f, fw / "input.wav")
        if isolate:
            cur, _ = separate.split_vocals(cur, fw)
            if lead_only:
                cur, _ = separate.split_backing(cur, fw)
        if dereverb:
            cur = separate.dereverb(cur, fw / "dry.wav")
        if denoise:
            cur = separate.denoise(cur, fw / "clean.wav")
        to_wav(cur, target, sr=44100, mono=True)
    shutil.rmtree(work, ignore_errors=True)
    total = sum(duration_seconds(f) for f in out.glob("*.wav"))
    log(f"Dataset '{name}': {len(list(out.glob('*.wav')))} files, {total / 60:.1f} minutes -> {out}")
    if total < 5 * 60:
        log("WARNING: under 5 minutes of audio. 10–30 minutes of clean singing gives far more natural results.")
    return out


# ----------------------------------------------------------------------- cover
def make_cover(
    song: str,
    voice: str,
    engine: str = "rvc",
    pitch: int = 0,
    key: int = 0,
    backing: str = "keep",
    dereverb: bool = True,
    index_rate: float = 0.5,
    protect: float = 0.33,
    f0_method: str = "rmvpe",
    reverb: float = 0.25,
    vocal_gain_db: float = 0.0,
    diffusion_steps: int = 40,
    out_name: str | None = None,
) -> Path:
    """Song in, cover out.

    pitch: semitones applied to the *vocal only* (e.g. +12 / -12 for male<->female
           while staying in the original key).
    key:   semitones applied to the *whole song* (vocal + music) to move it into
           a key that suits the new singer.
    backing: 'keep' original backing vocals, 'convert' them too, or 'drop' them.
    """
    paths = config.get_paths()
    song_path = Path(song).expanduser()
    if not song_path.exists():
        raise AISingerError(f"Song not found: {song_path}")
    if backing not in ("keep", "convert", "drop"):
        raise AISingerError("backing must be keep, convert or drop")
    eng = get_engine(engine)
    slug = slugify(song_path.stem)
    voice_slug = slugify(Path(voice).stem)
    work = paths.outputs / slug / "stems"
    work.mkdir(parents=True, exist_ok=True)

    log("1/5 Separating vocals from music ...")
    src = to_wav(song_path, work / "song.wav") if not (work / "song.wav").exists() else work / "song.wav"
    vocals, inst = separate.split_vocals(src, work)

    log("2/5 Splitting lead vs backing vocals ...")
    lead, back = separate.split_backing(vocals, work)

    if dereverb:
        log("3/5 Removing reverb/echo from the lead vocal ...")
        lead_dry = separate.dereverb(lead, work / "lead_dry.wav")
    else:
        lead_dry = lead

    log(f"4/5 Converting vocal to '{voice}' with {engine} ...")
    total_pitch = pitch + key
    conv_dir = paths.outputs / slug / voice_slug
    conv_dir.mkdir(parents=True, exist_ok=True)
    converted = conv_dir / "vocal_converted.wav"
    kwargs = dict(pitch=total_pitch)
    if engine.lower() in ("rvc", "applio"):
        kwargs.update(index_rate=index_rate, protect=protect, f0_method=f0_method)
    else:
        kwargs.update(diffusion_steps=diffusion_steps)
    eng.convert(lead_dry, converted, voice, **kwargs)

    backing_track = None
    if back is not None and backing == "convert":
        back_conv = conv_dir / "backing_converted.wav"
        eng.convert(back, back_conv, voice, **kwargs)
        backing_track = back_conv
    elif back is not None and backing == "keep":
        backing_track = mix.pitch_shift(back, conv_dir / "backing_keyed.wav", key)

    log("5/5 Mixing ...")
    inst_keyed = mix.pitch_shift(inst, conv_dir / "instrumental_keyed.wav", key)
    final_name = out_name or f"{slug}__{voice_slug}"
    final_wav = mix.mix(
        converted,
        inst_keyed,
        paths.outputs / slug / f"{final_name}.wav",
        backing=backing_track,
        match_to=lead_dry,
        vocal_gain_db=vocal_gain_db,
        reverb=reverb,
    )
    final_mp3 = to_mp3(final_wav, final_wav.with_suffix(".mp3"))
    log(f"Done! {final_mp3}")
    log(f"Converted vocal only: {converted}")
    return final_mp3


# --------------------------------------------------------------------- convert
def convert_vocal(
    input_path: str,
    voice: str,
    engine: str = "rvc",
    pitch: int = 0,
    clean: bool = False,
    out: str | None = None,
    **kwargs,
) -> Path:
    """Convert a vocal-only recording (e.g. you singing) into another voice."""
    paths = config.get_paths()
    src = Path(input_path).expanduser()
    if not src.exists():
        raise AISingerError(f"Not found: {src}")
    slug = slugify(src.stem)
    work = paths.outputs / "converted" / slug
    cur = to_wav(src, work / "input.wav")
    if clean:
        cur = separate.dereverb(cur, work / "dry.wav")
    dst = Path(out) if out else work / f"{slug}__{slugify(Path(voice).stem)}.wav"
    get_engine(engine).convert(cur, dst, voice, pitch=pitch, **kwargs)
    log(f"Done! {dst}")
    return dst
