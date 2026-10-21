import csv

import pytest
from click.testing import CliRunner

from moment_miner.cli import main
from moment_miner.compare import agreement, balanced_order, build, tokens


def rows(n, text, video="v"):
    return [{"id": f"{video}:{i * 4.0}", "video_id": video, "t0": i * 4.0,
             "t1": i * 4.0 + 8.0, "caption": text(i)} for i in range(n)]


def cache(data_dir, name, n, text):
    from moment_miner.captions import CaptionCache, parse_axes
    sc = CaptionCache(data_dir, name)
    for r in rows(n, text):
        sc.put(r["id"], r["t0"], r["t1"], parse_axes(r["caption"].replace(",", " ")), "m")


# --- the agreement signal -------------------------------------------------

def test_tokens_drops_stopwords_and_plurals():
    assert tokens("a person vaults over the rails") == {"person", "vault", "rail"}


def test_agreement_finds_shared_and_unique_terms():
    a = agreement(["kong vault over a rail, concrete plaza",
                   "kong vault over a rail, empty plaza"])
    assert "vault" in a["shared"] and "rail" in a["shared"]
    assert a["unique"][0] == ["concrete"]
    assert a["unique"][1] == ["empty"]
    assert not a["disagree"]


def test_disagreement_flagged_when_nothing_is_shared():
    a = agreement(["pull-up on a bar, outdoor gym", "waves breaking on a beach"])
    assert a["shared"] == []
    assert a["disagree"]
    assert a["min_overlap"] == 0.0


def test_one_shared_word_is_enough_to_not_flag():
    """The mark is 'no word in common', not a graded score: a tuned threshold
    fired on 17 of 18 real segments and carried no information."""
    a = agreement(["man climbs a sea cliff above crashing waves at sunset",
                   "man stands on rocks"])
    assert a["shared"] == ["man"]
    assert not a["disagree"]


def test_containment_not_jaccard_so_length_is_not_disagreement():
    a = agreement(["kong vault over a rail",
                   "kong vault over a rail in a wide concrete plaza at dusk with onlookers"])
    assert a["min_overlap"] == 1.0


def test_paraphrase_reads_as_disagreement():
    """Named so the limitation cannot be mistaken for a bug: the signal is
    lexical, so two right answers in different words look like a conflict."""
    assert agreement(["athlete bails a trick", "person fails a landing"])["disagree"]


def test_single_caption_set_never_disagrees():
    assert not agreement(["kong vault over a rail"])["disagree"]


# --- blinding -------------------------------------------------------------

def test_balanced_order_uses_every_slot_equally():
    deck = balanced_order(18, 3, seed=1)
    for slot in range(3):
        counts = [sum(1 for row in deck if row[slot] == s) for s in range(3)]
        assert counts == [6, 6, 6]


def test_blind_page_hides_the_set_names():
    sets = {f"model{n}": rows(6, lambda i, n=n: f"caption {n} for {i}") for n in (1, 2)}
    html, order = build(sets, blind=True, seed=1, frames=False)
    assert "model1" not in html and "model2" not in html
    assert len(order) == 6
    assert {"A", "B"} <= set(order[0])


def test_no_blind_shows_the_set_names():
    sets = {f"model{n}": rows(6, lambda i, n=n: f"caption {n} for {i}") for n in (1, 2)}
    html, _ = build(sets, blind=False, frames=False)
    assert "model1" in html and "model2" in html


# --- inputs ---------------------------------------------------------------

def test_sets_with_no_segment_in_common_are_an_error():
    with pytest.raises(ValueError, match="captioned in every set"):
        build({"a": rows(3, lambda i: "x"), "b": rows(3, lambda i: "y", video="w")},
              frames=False)


def test_only_segments_present_in_every_set_are_compared():
    _, order = build({"a": rows(6, lambda i: "x"), "b": rows(4, lambda i: "y")}, frames=False)
    assert len(order) == 4


# --- the command ----------------------------------------------------------

def test_cli_writes_page_and_key(tmp_path):
    for n in (1, 2):
        cache(tmp_path / "data", f"m{n}", 6, lambda i, n=n: f"caption {n} {i}")
    out = tmp_path / "cmp"
    result = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "data"), "caption-compare", "m1", "m2",
        "-o", str(out), "--seed", "3",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "caption-comparison.html").exists()
    key = out / "caption-comparison-key.csv"
    rows_ = list(csv.DictReader(key.open()))
    assert len(rows_) == 6 and set(rows_[0]) == {"seg", "A", "B"}
    assert {rows_[0]["A"], rows_[0]["B"]} == {"m1", "m2"}


def test_cli_no_blind_writes_no_key(tmp_path):
    for n in (1, 2):
        cache(tmp_path / "data", f"m{n}", 6, lambda i, n=n: f"caption {n} {i}")
    out = tmp_path / "cmp"
    result = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "data"), "caption-compare", "m1", "m2",
        "-o", str(out), "--no-blind",
    ])
    assert result.exit_code == 0, result.output
    assert [p.name for p in out.iterdir()] == ["caption-comparison.html"]


def test_cli_names_a_set_that_is_not_in_the_cache(tmp_path):
    cache(tmp_path / "data", "m1", 6, lambda i: "x")
    result = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "data"), "caption-compare", "m1", "nope"])
    assert result.exit_code != 0 and "no captions in set 'nope'" in result.output


def test_frames_dir_is_used_when_the_archive_is_offline(tmp_path):
    import base64

    stills = tmp_path / "frames"
    stills.mkdir()
    blob = b"\xff\xd8\xff\xdb-stand-in-for-a-jpeg"
    for shot in "abc":
        (stills / f"v:0.0_clip_{shot}.jpg").write_bytes(blob)
    sets = {f"m{n}": rows(6, lambda i, n=n: f"caption {n} {i}") for n in (1, 2)}
    html, _ = build(sets, frames_dir=stills, blind=False)
    assert html.count(base64.standard_b64encode(blob).decode()) == 3


def test_frames_dir_matches_only_its_own_segment(tmp_path):
    stills = tmp_path / "frames"
    stills.mkdir()
    (stills / "s0_clip_a.jpg").write_bytes(b"zero")
    (stills / "s1_clip_a.jpg").write_bytes(b"one")
    from moment_miner.compare import load_frames
    assert len(load_frames(stills, "s0")) == 1


def test_page_defaults_to_the_data_folder(tmp_path):
    for n in (1, 2):
        cache(tmp_path / "data", f"m{n}", 6, lambda i, n=n: f"caption {n} {i}")
    result = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "data"), "caption-compare", "m1", "m2"])
    assert result.exit_code == 0, result.output
    [run] = (tmp_path / "data" / "cuts").glob("caption-compare_*")
    assert sorted(p.name for p in run.iterdir()) == ["caption-comparison-key.csv",
                                                     "caption-comparison.html"]


def test_caption_export_writes_the_set_as_csv(tmp_path):
    cache(tmp_path / "data", "m1", 3, lambda i: f"caption {i}")
    result = CliRunner().invoke(main, ["--data-dir", str(tmp_path / "data"),
                                       "caption-export", "m1", "-o", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    rows_ = list(csv.DictReader((tmp_path / "out" / "m1.csv").open()))
    assert [r["id"] for r in rows_] == ["v:0.0", "v:4.0", "v:8.0"]
    status = CliRunner().invoke(main, ["--data-dir", str(tmp_path / "data"), "status"])
    assert "captions: 3 in set 'm1'" in status.output
