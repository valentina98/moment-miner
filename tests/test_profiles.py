import pytest
from click.testing import CliRunner

from moment_miner.cli import _option_names, main
from moment_miner.profiles import load_profile, profile_names


def _mm(data_dir, *args):
    return CliRunner().invoke(main, ["--data-dir", str(data_dir), *args])


def _own(data_dir, name, text):
    (data_dir / "profiles").mkdir(parents=True, exist_ok=True)
    (data_dir / "profiles" / f"{name}.yaml").write_text(text)


def test_parkour_ships_and_every_key_is_an_option_of_its_command():
    assert "parkour" in profile_names()
    for command, keys in load_profile("parkour", "/nonexistent").items():
        assert set(keys) <= set(_option_names(main.commands[command])), command


def test_parkour_leaves_out_losslesscut_exports(tmp_path, monkeypatch):
    from moment_miner import cli

    monkeypatch.setattr(cli, "_report_stale_elsewhere", lambda *a: None)
    monkeypatch.setattr("moment_miner.indexer.index_pending",
                        lambda *a, **kw: {"indexed": 0, "errors": 0, "segments": 0})
    footage = tmp_path / "footage"
    footage.mkdir()
    for name in ("2M4A8050.MP4", "2M4A8050-00.00.23.993-00.00.28.358.MP4",
                 "2M4A8050-merged-1760000000000.MP4"):
        (footage / name).write_bytes(b"x" * 64)
    r = _mm(tmp_path / "data", "index", str(footage), "--backend", "mock", "--no-estimate",
            "--profile", "parkour")
    assert r.exit_code == 0, r.output
    assert "skipped: 2 video(s) matching --skip" in r.output


def test_every_command_but_help_takes_a_profile():
    for name, cmd in main.commands.items():
        assert ("--profile" in {o for p in cmd.params for o in p.opts}) == (name != "help"), name


def test_a_profile_sets_defaults_and_the_command_line_wins(tmp_path, monkeypatch):
    from moment_miner import cli

    seen = {}
    monkeypatch.setattr(cli, "_report_stale_elsewhere", lambda *a: None)

    def fake_index_pending(manifest, store, be, **kw):
        seen.update(win=kw["win"], stride=kw["stride"], asr=kw["use_asr"])
        return {"indexed": 0, "errors": 0, "segments": 0}

    monkeypatch.setattr("moment_miner.indexer.index_pending", fake_index_pending)
    _own(tmp_path / "data", "wide", "index:\n  window: 12\n  stride: 6\n  asr: false\n")
    footage = tmp_path / "footage"
    footage.mkdir()
    # `mm index` refuses a folder with no videos, as an unmounted drive.
    (footage / "a.mp4").write_bytes(b"x" * 64)
    base = ["index", str(footage), "--backend", "mock", "--no-estimate"]
    assert _mm(tmp_path / "data", *base, "--profile", "wide").exit_code == 0
    assert seen == {"win": 12.0, "stride": 6.0, "asr": False}
    assert _mm(tmp_path / "data", *base, "--profile", "wide", "--stride", "3").exit_code == 0
    assert seen == {"win": 12.0, "stride": 3.0, "asr": False}
    assert _mm(tmp_path / "data", *base, "--no-asr").exit_code == 0
    assert seen == {"win": 8.0, "stride": 4.0, "asr": False}


def test_the_users_own_profile_wins_over_a_shipped_one_of_the_same_name(tmp_path):
    _own(tmp_path / "data", "parkour", "annotate:\n  template: wedding\n")
    assert load_profile("parkour", tmp_path / "data") == {"annotate": {"template": "wedding"}}


def test_a_profile_file_can_be_named_by_path(tmp_path):
    p = tmp_path / "mine.yaml"
    p.write_text("search:\n  pad: 2.5\n")
    assert load_profile(str(p), tmp_path / "data") == {"search": {"pad": 2.5}}


@pytest.mark.parametrize("text, message", [
    ("index:\n  strid: 4\n", "has no option strid"),
    ("indx:\n  stride: 4\n", "'indx' is not a command"),
    ("index: 4\n", "one section per command"),
    ("index: [\n", "not valid YAML"),
])
def test_a_wrong_profile_fails_naming_the_fix(tmp_path, text, message):
    _own(tmp_path / "data", "bad", text)
    result = _mm(tmp_path / "data", "status", "--profile", "bad")
    assert result.exit_code == 2 and message in result.output, result.output


def test_an_unknown_profile_names_the_shipped_ones(tmp_path):
    result = _mm(tmp_path / "data", "status", "--profile", "nope")
    assert result.exit_code == 2 and "parkour" in result.output


def test_a_profile_adds_the_axes_its_caption_prompt_asks_beyond_the_default():
    from moment_miner.captions import profile_axes

    assert profile_axes("general") == ()
    assert profile_axes("parkour") == ("move",)
    assert "move" in profile_axes()


def test_only_the_searches_take_several_profiles(tmp_path):
    _own(tmp_path / "data", "wide", "index:\n  stride: 2.0\n")
    several = _mm(tmp_path / "data", "status", "--profile", "parkour,wide")
    assert several.exit_code == 2 and "takes one profile" in several.output
    unknown = _mm(tmp_path / "data", "search", "kong", "--profile", "parkour,nosuch")
    assert unknown.exit_code == 2 and "unknown profile 'nosuch'" in unknown.output
