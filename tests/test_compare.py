import csv

import pytest
from click.testing import CliRunner

from moment_miner.cli import main
from moment_miner.compare import agreement, balanced_order, build, tokens


def write_captions(path, rows, schema="sidecar"):
    with path.open("w", newline="") as f:
        if schema == "sidecar":
            w = csv.DictWriter(f, fieldnames=["id", "t0", "t1", "caption"], lineterminator="\n")
        else:
            w = csv.DictWriter(f, fieldnames=["seg", "caption"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def sidecar(n, text):
    return [{"id": f"s{i}", "t0": i * 4.0, "t1": i * 4.0 + 8.0, "caption": text(i)}
            for i in range(n)]


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


def test_blind_page_hides_the_file_names(tmp_path):
    for n in (1, 2):
        write_captions(tmp_path / f"model{n}.csv", sidecar(6, lambda i: f"caption {n} for {i}"))
    html, order = build([tmp_path / "model1.csv", tmp_path / "model2.csv"],
                        blind=True, seed=1, frames=False)
    assert "model1" not in html and "model2" not in html
    assert len(order) == 6
    assert {"A", "B"} <= set(order[0])


def test_no_blind_shows_the_file_names(tmp_path):
    for n in (1, 2):
        write_captions(tmp_path / f"model{n}.csv", sidecar(6, lambda i: f"caption {n} for {i}"))
    html, _ = build([tmp_path / "model1.csv", tmp_path / "model2.csv"],
                    blind=False, frames=False)
    assert "model1" in html and "model2" in html


# --- inputs ---------------------------------------------------------------

def test_bare_seg_caption_schema_is_accepted(tmp_path):
    for n in (1, 2):
        write_captions(tmp_path / f"m{n}.csv",
                       [{"seg": f"seg{i:02d}", "caption": f"c{n}{i}"} for i in range(6)],
                       schema="bare")
    html, order = build([tmp_path / "m1.csv", tmp_path / "m2.csv"], blind=True, frames=False)
    assert len(order) == 6
    assert "seg01" in html


def test_segments_with_no_overlap_are_an_error(tmp_path):
    write_captions(tmp_path / "a.csv", sidecar(3, lambda i: "x"))
    write_captions(tmp_path / "b.csv",
                   [{"id": f"other{i}", "t0": 0.0, "t1": 8.0, "caption": "y"} for i in range(3)])
    with pytest.raises(ValueError, match="every caption file"):
        build([tmp_path / "a.csv", tmp_path / "b.csv"], frames=False)


def test_only_segments_present_in_all_files_are_compared(tmp_path):
    write_captions(tmp_path / "a.csv", sidecar(6, lambda i: "x"))
    write_captions(tmp_path / "b.csv", sidecar(4, lambda i: "y"))
    _, order = build([tmp_path / "a.csv", tmp_path / "b.csv"], frames=False)
    assert len(order) == 4


# --- the command ----------------------------------------------------------

def test_cli_writes_page_and_key(tmp_path):
    for n in (1, 2):
        write_captions(tmp_path / f"m{n}.csv", sidecar(6, lambda i: f"caption {n} {i}"))
    out = tmp_path / "cmp.html"
    result = CliRunner().invoke(main, [
        "caption-compare", str(tmp_path / "m1.csv"), str(tmp_path / "m2.csv"),
        "-o", str(out), "--seed", "3",
    ])
    assert result.exit_code == 0, result.output
    assert out.exists()
    key = tmp_path / "cmp-key.csv"
    assert key.exists()
    rows = list(csv.DictReader(key.open()))
    assert len(rows) == 6 and set(rows[0]) == {"seg", "A", "B"}


def test_cli_no_blind_writes_no_key(tmp_path):
    for n in (1, 2):
        write_captions(tmp_path / f"m{n}.csv", sidecar(6, lambda i: f"caption {n} {i}"))
    out = tmp_path / "cmp.html"
    result = CliRunner().invoke(main, [
        "caption-compare", str(tmp_path / "m1.csv"), str(tmp_path / "m2.csv"),
        "-o", str(out), "--no-blind",
    ])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "cmp-key.csv").exists()


def test_frames_dir_is_used_when_the_archive_is_offline(tmp_path):
    import base64

    stills = tmp_path / "frames"
    stills.mkdir()
    blob = b"\xff\xd8\xff\xdb-stand-in-for-a-jpeg"
    for shot in "abc":
        (stills / f"s0_clip_{shot}.jpg").write_bytes(blob)
    for n in (1, 2):
        write_captions(tmp_path / f"m{n}.csv", sidecar(6, lambda i: f"caption {n} {i}"))
    html, _ = build([tmp_path / "m1.csv", tmp_path / "m2.csv"],
                    frames_dir=stills, blind=False)
    assert html.count(base64.standard_b64encode(blob).decode()) == 3


def test_frames_dir_matches_only_its_own_segment(tmp_path):
    stills = tmp_path / "frames"
    stills.mkdir()
    (stills / "s0_clip_a.jpg").write_bytes(b"zero")
    (stills / "s1_clip_a.jpg").write_bytes(b"one")
    from moment_miner.compare import load_frames
    assert len(load_frames(stills, "s0")) == 1


def test_page_defaults_next_to_the_footage(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    for n in (1, 2):
        write_captions(tmp_path / f"m{n}.csv", sidecar(6, lambda i: f"caption {n} {i}"))
    result = CliRunner().invoke(main, [
        "caption-compare", str(tmp_path / "m1.csv"), str(tmp_path / "m2.csv"),
        "--videos", str(archive),
    ])
    assert result.exit_code == 0, result.output
    assert (archive / "caption-comparison.html").exists()
    assert (archive / "caption-comparison-key.csv").exists()
    assert not (tmp_path / "caption-comparison.html").exists()
