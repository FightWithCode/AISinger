"""Fast tests that need no GPU, models or engines:  python -m pytest tests/"""

import numpy as np
import pytest

from aisinger import cli, separate


@pytest.fixture(autouse=True)
def _tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AISINGER_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AISINGER_CACHE", str(tmp_path / "cache"))


def test_parser_cover_defaults():
    a = cli.build_parser().parse_args(["cover", "song.mp3", "--voice", "me"])
    assert (a.engine, a.pitch, a.key, a.backing, a.index_rate) == ("rvc", 0, 0, "keep", 0.5)


def test_stem_name_parsing():
    assert separate._stem_name("song_(Vocals)_model_bs_roformer.wav") == "vocals"
    assert separate._stem_name("x_(No Reverb)_UVR-DeEcho-DeReverb.wav") == "no reverb"


def test_pick_prefers_no_reverb(tmp_path):
    stems = {"reverb": tmp_path / "a.wav", "no reverb": tmp_path / "b.wav"}
    assert separate._pick(stems, "no reverb", "noreverb") == tmp_path / "b.wav"


def test_find_model_picks_latest_epoch_and_pin(tmp_path):
    from aisinger.config import get_paths
    from aisinger.engines.applio import Applio

    vdir = get_paths().voice_dir("me")
    vdir.mkdir(parents=True)
    for n in ["G_2333333.pth", "D_2333333.pth", "me_25e_100s.pth", "me_300e_1200s.pth", "me_75e_300s.pth"]:
        (vdir / n).write_bytes(b"x")
    (vdir / "me.index").write_bytes(b"x")
    pth, idx = Applio().find_model("me")
    assert pth.name == "me_300e_1200s.pth" and idx.name == "me.index"
    cli.main(["use", "me", "me_75e_300s.pth"])
    assert Applio().find_model("me")[0].name == "me_75e_300s.pth"


def test_mix_does_not_clip(tmp_path):
    from aisinger import mix

    t = np.linspace(0, 3, 3 * mix.SR, dtype=np.float32)
    loud = np.stack([np.sin(2 * np.pi * 220 * t)] * 2) * 0.99
    mix.save(tmp_path / "v.wav", loud)
    mix.save(tmp_path / "i.wav", loud)
    out = mix.mix(tmp_path / "v.wav", tmp_path / "i.wav", tmp_path / "o.wav")
    assert np.abs(mix.load(out)).max() <= 0.96
