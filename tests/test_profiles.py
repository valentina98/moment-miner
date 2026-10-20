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
