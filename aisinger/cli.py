"""Command-line interface: ``aisinger <command> ...``  (run ``aisinger -h``)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config
from .utils import AISingerError, duration_seconds, log

EXAMPLES = """
examples:
  aisinger setup                                    # install engines (once per machine/Colab session)
  aisinger dataset me ~/my_singing/                  # prepare your recordings
  aisinger train me                                  # clone your voice (RVC)
  aisinger cover "O Mere Dil Ke Chain.mp3" --voice me --key -2
  aisinger dataset kishore songs/ --isolate --lead-only --dereverb
  aisinger reference kishore clean_kishore_clip.wav  # zero-shot voice (no training)
  aisinger cover rafi_song.mp3 --voice kishore --engine seedvc
  aisinger speak "Namaste, yeh meri awaaz hai" --voice me
"""


def _add_conversion_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--voice", required=True, help="voice name (folder in voices/), a .pth file, or (seedvc) a reference clip")
    p.add_argument("--engine", default="rvc", choices=["rvc", "seedvc"], help="rvc = trained model (best quality); seedvc = zero-shot from a clip")
    p.add_argument("--pitch", type=int, default=0, help="semitones for the vocal only (+12/-12 for female<->male)")
    p.add_argument("--index-rate", type=float, default=0.5, help="[rvc] 0–1, how strongly to apply the voice's accent/timbre index")
    p.add_argument("--protect", type=float, default=0.33, help="[rvc] 0–0.5, protects breaths/consonants (lower = stronger)")
    p.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "fcpe", "crepe", "hybrid[crepe+rmvpe]"], help="[rvc] pitch tracker")
    p.add_argument("--steps", type=int, default=40, help="[seedvc] diffusion steps (more = better, slower)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="aisinger",
        description="Make any voice sing any song.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--version", action="version", version=f"aisinger {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="install voice engines")
    p.add_argument("--engine", default="all", choices=["all", "rvc", "seedvc"])
    p.add_argument("--cpu", action="store_true", help="install CPU-only PyTorch for engines")
    p.add_argument("--force", action="store_true", help="reinstall")

    sub.add_parser("doctor", help="show paths, GPU and installed engines")
    sub.add_parser("voices", help="list voices")

    p = sub.add_parser("separate", help="split a song into stems")
    p.add_argument("song")
    p.add_argument("--no-backing-split", action="store_true")
    p.add_argument("--no-dereverb", action="store_true")

    p = sub.add_parser("dataset", help="prepare training audio for a voice")
    p.add_argument("name")
    p.add_argument("inputs", nargs="+", help="audio files or folders")
    p.add_argument("--isolate", action="store_true", help="inputs are full songs: extract the vocals first")
    p.add_argument("--lead-only", action="store_true", help="with --isolate: also remove backing vocals/chorus")
    p.add_argument("--dereverb", action="store_true", help="remove reverb/echo (recommended for songs)")
    p.add_argument("--denoise", action="store_true", help="remove background noise")

    p = sub.add_parser("train", help="train (clone) a voice")
    p.add_argument("name")
    p.add_argument("--data", help="dataset folder (default: datasets/<name>)")
    p.add_argument("--engine", default="rvc", choices=["rvc", "seedvc"])
    p.add_argument("--epochs", type=int, default=300, help="[rvc] total epochs (200–500 typical)")
    p.add_argument("--save-every", type=int, default=25, help="[rvc] save a usable model every N epochs")
    p.add_argument("--batch-size", type=int, default=None, help="default: 8 for rvc, 2 for seedvc")
    p.add_argument("--sample-rate", type=int, default=40000, choices=[32000, 40000, 48000], help="[rvc]")
    p.add_argument("--steps", type=int, default=1000, help="[seedvc] fine-tuning steps")
    p.add_argument("--fresh", action="store_true", help="discard previous training of this voice and start over")

    p = sub.add_parser("import", help="import an existing RVC model (.pth/.index, folder, .zip or URL)")
    p.add_argument("name")
    p.add_argument("source")

    p = sub.add_parser("use", help="[rvc] choose which saved epoch a voice uses (if the latest sounds over-trained)")
    p.add_argument("name")
    p.add_argument("pth", nargs="?", help="file name of the .pth to use; omit to list / reset to latest")

    p = sub.add_parser("reference", help="[seedvc] set the reference clip for a zero-shot voice")
    p.add_argument("name")
    p.add_argument("clip", help="5–25 s of clean, dry singing/speech of the target voice")
    p.add_argument("--start", type=float, default=0.0, help="start offset in seconds")
    p.add_argument("--isolate", action="store_true", help="clip has music: extract + de-reverb the vocal first")

    p = sub.add_parser("cover", help="make a song sung by a different voice")
    p.add_argument("song")
    _add_conversion_opts(p)
    p.add_argument("--key", type=int, default=0, help="transpose the whole song (vocal + music) by N semitones")
    p.add_argument("--backing", default="keep", choices=["keep", "convert", "drop"], help="what to do with backing vocals/chorus")
    p.add_argument("--no-dereverb", action="store_true", help="skip removing reverb before conversion")
    p.add_argument("--reverb", type=float, default=0.25, help="reverb added back on the new vocal (0–1)")
    p.add_argument("--vocal-gain", type=float, default=0.0, help="dB, adjust vocal loudness in the mix")
    p.add_argument("--out-name", help="output file name (without extension)")

    p = sub.add_parser("convert", help="convert a vocal-only recording into another voice")
    p.add_argument("input")
    _add_conversion_opts(p)
    p.add_argument("--dereverb", action="store_true", help="remove room reverb first")
    p.add_argument("-o", "--output")

    p = sub.add_parser("speak", help="[rvc] text-to-speech in a cloned voice")
    p.add_argument("text")
    p.add_argument("--voice", required=True)
    p.add_argument("--tts-voice", default="hi-IN-MadhurNeural", help="base TTS voice, e.g. hi-IN-MadhurNeural, hi-IN-SwaraNeural, en-IN-PrabhatNeural")
    p.add_argument("--rate", type=int, default=0, help="speaking rate -100..100")
    p.add_argument("--pitch", type=int, default=0)
    p.add_argument("-o", "--output")
    return ap


def _conv_kwargs(a) -> dict:
    if a.engine == "rvc":
        return dict(index_rate=a.index_rate, protect=a.protect, f0_method=a.f0_method)
    return dict(diffusion_steps=a.steps)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    paths = config.get_paths()

    if a.cmd == "setup":
        from .engines.applio import Applio
        from .engines.seedvc import SeedVC

        gpu = False if a.cpu else None
        if a.engine in ("all", "rvc"):
            Applio().install(gpu=gpu, force=a.force)
        if a.engine in ("all", "seedvc"):
            SeedVC().install(gpu=gpu, force=a.force)

    elif a.cmd == "doctor":
        from .engines.applio import Applio
        from .engines.seedvc import SeedVC

        print(f"aisinger {__version__}")
        print(f"AISINGER_HOME  = {paths.home}")
        print(f"AISINGER_CACHE = {paths.cache}")
        print(f"NVIDIA GPU     = {'yes' if config.has_nvidia_gpu() else 'no (CPU only: training will be very slow)'}")
        print(f"RVC (Applio)   = {'installed' if Applio().installed else 'not installed'}")
        print(f"Seed-VC        = {'installed' if SeedVC().installed else 'not installed'}")

    elif a.cmd == "voices":
        _list_voices(paths)

    elif a.cmd == "separate":
        from . import separate
        from .utils import slugify, to_wav

        song = Path(a.song).expanduser()
        work = paths.outputs / slugify(song.stem) / "stems"
        src = to_wav(song, work / "song.wav")
        vocals, inst = separate.split_vocals(src, work)
        if not a.no_backing_split:
            lead, _ = separate.split_backing(vocals, work)
            if not a.no_dereverb:
                separate.dereverb(lead, work / "lead_dry.wav")
        log(f"Stems in {work}")

    elif a.cmd == "dataset":
        from .pipeline import prepare_dataset

        prepare_dataset(a.name, a.inputs, a.isolate, a.lead_only, a.dereverb, a.denoise)

    elif a.cmd == "train":
        from .pipeline import get_engine

        data = Path(a.data).expanduser() if a.data else paths.datasets / a.name
        if not data.exists() or not any(data.iterdir()):
            raise AISingerError(f"No dataset at {data}. Run: aisinger dataset {a.name} <your audio files>")
        eng = get_engine(a.engine)
        if a.engine == "rvc":
            eng.train(
                a.name, data, epochs=a.epochs, save_every=a.save_every,
                batch_size=a.batch_size or 8, sample_rate=a.sample_rate, fresh=a.fresh,
            )
        else:
            eng.train(a.name, data, steps=a.steps, batch_size=a.batch_size or 2, fresh=a.fresh)

    elif a.cmd == "import":
        from .engines.applio import Applio

        Applio().import_model(a.name, a.source)

    elif a.cmd == "use":
        vdir = paths.voice_dir(a.name)
        pths = sorted(f.name for f in vdir.glob("*.pth") if not f.name.startswith(("G_", "D_")))
        if not a.pth:
            (vdir / "use.txt").unlink(missing_ok=True)
            print("Saved models (now using the latest):\n  " + "\n  ".join(pths))
        else:
            if a.pth not in pths:
                raise AISingerError(f"{a.pth} not found. Available:\n  " + "\n  ".join(pths))
            (vdir / "use.txt").write_text(a.pth)
            log(f"Voice '{a.name}' now uses {a.pth}")

    elif a.cmd == "reference":
        from .engines.seedvc import SeedVC

        clip = Path(a.clip).expanduser()
        if a.isolate:
            from . import separate
            from .utils import to_wav

            work = paths.cache / "reference-work" / a.name
            src = to_wav(clip, work / "clip.wav")
            vocals, _ = separate.split_vocals(src, work)
            lead, _ = separate.split_backing(vocals, work)
            clip = separate.dereverb(lead, work / "dry.wav")
        SeedVC().set_reference(a.name, clip, start=a.start)

    elif a.cmd == "cover":
        from .pipeline import make_cover

        make_cover(
            a.song, a.voice, engine=a.engine, pitch=a.pitch, key=a.key, backing=a.backing,
            dereverb=not a.no_dereverb, index_rate=a.index_rate, protect=a.protect,
            f0_method=a.f0_method, reverb=a.reverb, vocal_gain_db=a.vocal_gain,
            diffusion_steps=a.steps, out_name=a.out_name,
        )

    elif a.cmd == "convert":
        from .pipeline import convert_vocal

        convert_vocal(a.input, a.voice, engine=a.engine, pitch=a.pitch, clean=a.dereverb, out=a.output, **_conv_kwargs(a))

    elif a.cmd == "speak":
        from .engines.applio import Applio
        from .utils import slugify

        out = Path(a.output) if a.output else paths.outputs / "speech" / f"{slugify(a.voice)}_{slugify(a.text[:30])}.wav"
        Applio().speak(a.text, a.voice, out, tts_voice=a.tts_voice, tts_rate=a.rate, pitch=a.pitch)
        log(f"Done! {out}")
    return 0


def _list_voices(paths: config.Paths) -> None:
    rows = []
    for d in sorted(p for p in paths.voices.iterdir() if p.is_dir()):
        pths = [f for f in d.glob("*.pth") if not f.name.startswith(("G_", "D_"))]
        kinds = []
        if pths:
            kinds.append(f"rvc ({len(pths)} saved epoch{'s' if len(pths) != 1 else ''})")
        if (d / "reference.wav").exists():
            ft = " + fine-tuned" if (d / "seedvc_run" / "ft_model.pth").exists() else ""
            kinds.append(f"seedvc (ref {duration_seconds(d / 'reference.wav'):.0f}s{ft})")
        if any(d.glob("G_*.pth")) and not pths:
            kinds.append("training in progress")
        if kinds:
            rows.append(f"  {d.name:<20} {', '.join(kinds)}")
    print("Voices in", paths.voices)
    print("\n".join(rows) if rows else "  (none yet)")


def entrypoint() -> None:
    try:
        sys.exit(main())
    except AISingerError as e:
        print(f"\n[aisinger] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    entrypoint()
