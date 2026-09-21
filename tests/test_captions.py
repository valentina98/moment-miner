import csv
import json
import types
from pathlib import Path

import numpy as np
import pytest

from moment_miner.captions import (
    CaptionSidecar,
    even_points,
    frames_for_duration,
    sample_points,
    ClaudeCaptionBackend,
    MissingCaptionCredentials,
    MockCaptionBackend,
    get_caption_backend,
    resolve_credentials,
    still_times,
    subsample,
    load_prompt,
    prompt_names,
)

from .conftest import requires_ffmpeg


def frames(n=8, h=48, w=64):
    return np.zeros((n, h, w, 3), dtype=np.uint8)


class FakeMessages:
    """Stands in for client.messages — records the request, returns one block."""

    def __init__(self, text="a person vaults a rail in a concrete plaza"):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = types.SimpleNamespace(type="text", text=self.text)
        return types.SimpleNamespace(content=[block])


def fake_client(text="a person vaults a rail in a concrete plaza"):
    return types.SimpleNamespace(messages=FakeMessages(text))


# --- credential resolution: the three paths -------------------------------

CLAUDE_CREDS = {"claudeAiOauth": {"accessToken": "oauth-value", "scopes": []}}
# Another service's token, placed first in the file under the same key name.
OTHER_CREDS = {"mcpOAuth": {"other|abc": {"accessToken": "not-claude"}}}


def test_other_token_is_never_used(tmp_path):
    (tmp_path / ".credentials.json").write_text(json.dumps({**OTHER_CREDS, **CLAUDE_CREDS}))
    assert resolve_credentials(env={}, config_dir=tmp_path) == {"auth_token": "oauth-value"}


def test_other_token_alone_is_no_credential(tmp_path):
    (tmp_path / ".credentials.json").write_text(json.dumps(OTHER_CREDS))
    with pytest.raises(MissingCaptionCredentials):
        resolve_credentials(env={}, config_dir=tmp_path)


def test_subscription_wins_over_api_key(tmp_path):
    """The free route is preferred: quota, not money, when both are available."""
    (tmp_path / ".credentials.json").write_text(json.dumps(CLAUDE_CREDS))
    got = resolve_credentials(env={"ANTHROPIC_API_KEY": "sk-test"}, config_dir=tmp_path)
    assert got == {"auth_token": "oauth-value"}


def test_api_key_refused_without_permission(tmp_path):
    """A key alone is not consent: no subscription and no --allow-paid means stop."""
    with pytest.raises(MissingCaptionCredentials) as e:
        resolve_credentials(env={"ANTHROPIC_API_KEY": "sk-test"}, config_dir=tmp_path)
    assert "--allow-paid" in str(e.value)


def test_api_key_refused_without_a_budget(tmp_path):
    with pytest.raises(MissingCaptionCredentials):
        resolve_credentials(env={"ANTHROPIC_API_KEY": "sk-test"}, config_dir=tmp_path,
                            allow_paid=True)


def test_api_key_refused_on_a_zero_budget(tmp_path):
    with pytest.raises(MissingCaptionCredentials):
        resolve_credentials(env={"ANTHROPIC_API_KEY": "sk-test"}, config_dir=tmp_path,
                            allow_paid=True, paid_budget_usd=0)


def test_api_key_used_when_allowed_with_a_budget(tmp_path):
    got = resolve_credentials(env={"ANTHROPIC_API_KEY": "sk-test"}, config_dir=tmp_path,
                              allow_paid=True, paid_budget_usd=5.0)
    assert got == {"api_key": "sk-test"}


def test_subscription_credentials_used_when_no_key(tmp_path):
    (tmp_path / ".credentials.json").write_text(json.dumps(CLAUDE_CREDS))
    assert resolve_credentials(env={}, config_dir=tmp_path) == {"auth_token": "oauth-value"}


def test_no_credentials_raises_naming_both(tmp_path):
    with pytest.raises(MissingCaptionCredentials) as e:
        resolve_credentials(env={}, config_dir=tmp_path)
    assert "ANTHROPIC_API_KEY" in str(e.value)
    assert "Claude Code" in str(e.value)


def test_oauth_route_sends_the_beta_header(tmp_path, monkeypatch):
    (tmp_path / ".credentials.json").write_text(json.dumps(CLAUDE_CREDS))
    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.messages = FakeMessages()

    monkeypatch.setitem(
        __import__("sys").modules, "anthropic",
        types.SimpleNamespace(Anthropic=FakeAnthropic),
    )
    be = ClaudeCaptionBackend("claude-haiku-4-5", env={"CLAUDE_CONFIG_DIR": str(tmp_path)})
    be.client
    assert captured["auth_token"] == "oauth-value"
    assert captured["default_headers"]["anthropic-beta"] == "oauth-2025-04-20"


# --- captioning -----------------------------------------------------------

def test_caption_sends_the_stills_the_rule_asks_for_and_returns_text():
    client = fake_client()
    be = ClaudeCaptionBackend("claude-haiku-4-5", client=client)
    out = be.caption(frames(8), span=8.0, stride=4.0)
    assert out == "a person vaults a rail in a concrete plaza"
    content = client.messages.calls[0]["messages"][0]["content"]
    assert sum(1 for b in content if b["type"] == "image") == 2
    assert content[-1]["type"] == "text"
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_subsample_never_invents_frames():
    assert len(subsample(frames(2), 8.0, 4.0)) == 2
    assert len(subsample(frames(3), 8.0, 4.0, n=8)) == 3


def test_sampling_follows_the_stride_in_three_tiers():
    assert sample_points(2.0, 4.0) == [0.5]                    # under one stride
    assert sample_points(4.0, 4.0) == [1 / 3, 2 / 3]           # under two strides
    pts = sample_points(8.0, 4.0)                              # the ordinary segment
    assert [round(p * 8, 1) for p in pts] == [2.0, 6.0]


def test_the_last_frame_sits_half_a_stride_from_the_end():
    """Anchored at the end, where the action resolves — not at the midpoint."""
    for span in (8.0, 12.0, 16.0):
        assert round(span - sample_points(span, 4.0)[-1] * span, 6) == 2.0


def test_one_frame_per_stride_block_at_its_midpoint():
    """The step is a whole stride, so with windows overlapping by half, two
    neighbouring windows agree on the frame they share instead of each adding
    one. Ordinary segments cost two frames rather than three."""
    assert [round(p * 8, 1) for p in sample_points(8.0, 4.0)] == [2.0, 6.0]
    assert [round(p * 12, 1) for p in sample_points(12.0, 4.0)] == [2.0, 6.0, 10.0]
    shared = round(sample_points(8.0, 4.0)[-1] * 8, 1)          # 6.0s into window t
    assert round(sample_points(8.0, 4.0)[0] * 8 + 4, 1) == shared


def test_a_finer_stride_multiplies_the_frame_count():
    """Documented fragility, asserted so it cannot change unnoticed: the tiers
    were reasoned about at stride 4 (see sample_points' docstring)."""
    assert len(sample_points(8.0, 2.0)) == 4


def test_an_explicit_count_overrides_the_rule():
    assert len(subsample(frames(8), 8.0, 4.0)) == 2
    assert len(subsample(frames(8), 8.0, 4.0, n=5)) == 5
    assert even_points(1) == [1.0]


def test_frames_for_duration_forces_a_count_per_tier():
    kw = dict(short=1, normal=3, long=6, short_video_s=8.0, long_video_s=120.0)
    assert frames_for_duration(4.2, **kw) == 1
    assert frames_for_duration(21.0, **kw) == 3
    assert frames_for_duration(300.0, **kw) == 6


def test_no_forced_count_means_derive_from_the_stride():
    assert frames_for_duration(4.2) is None
    assert frames_for_duration(300.0) is None
    assert frames_for_duration(300.0, short=1, normal=3, long=None) == 3


def test_a_short_clip_is_captioned_from_its_final_frame(monkeypatch):
    """A 4-second clip gets one still, and it is the last: in this footage the
    trick lands and recording stops."""
    client = fake_client()
    be = ClaudeCaptionBackend("claude-haiku-4-5", client=client)
    eight = frames(8)
    eight[-1] = 200
    be.caption(eight, span=3.0, stride=4.0)
    content = client.messages.calls[0]["messages"][0]["content"]
    assert sum(1 for b in content if b["type"] == "image") == 1
    assert len(subsample(eight, 3.0, 4.0)) == 1


def test_caption_sends_the_requested_number_of_frames():
    client = fake_client()
    be = ClaudeCaptionBackend("claude-haiku-4-5", client=client)
    be.caption(frames(8), span=8.0, n_frames=5)
    content = client.messages.calls[0]["messages"][0]["content"]
    assert sum(1 for b in content if b["type"] == "image") == 5


def test_preflight_raises_before_any_video_is_touched(tmp_path):
    be = ClaudeCaptionBackend("claude-haiku-4-5", env={"CLAUDE_CONFIG_DIR": str(tmp_path)})
    with pytest.raises(MissingCaptionCredentials):
        be.preflight()


def test_backend_selection():
    assert isinstance(get_caption_backend("mock"), MockCaptionBackend)
    assert get_caption_backend("claude-haiku-4-5").name == "claude-haiku-4-5"
    with pytest.raises(ValueError, match="unknown caption backend"):
        get_caption_backend("gpt-4")


# --- the sidecar beside the footage ---------------------------------------

def test_sidecar_writes_next_to_the_video(tmp_path):
    video = tmp_path / "clips" / "a.mp4"
    video.parent.mkdir()
    sc = CaptionSidecar(video)
    sc.put("a:0.0", 0.0, 8.0, "a caption", "claude-haiku-4-5")
    sc.flush()
    written = video.parent / "captions.csv"
    assert written.exists()
    rows = list(csv.DictReader(written.open()))
    assert rows[0]["id"] == "a:0.0"
    assert rows[0]["caption"] == "a caption"
    assert rows[0]["model"] == "claude-haiku-4-5"


def test_sidecar_is_read_back_so_captions_are_not_bought_twice(tmp_path):
    video = tmp_path / "a.mp4"
    sc = CaptionSidecar(video)
    sc.put("a:0.0", 0.0, 8.0, "cached", "claude-haiku-4-5")
    sc.flush()
    assert CaptionSidecar(video).get("a:0.0") == "cached"
    assert CaptionSidecar(video).get("a:4.0") is None


def test_caption_dir_override(tmp_path):
    video = tmp_path / "a.mp4"
    out = tmp_path / "elsewhere"
    sc = CaptionSidecar(video, caption_dir=out)
    sc.put("a:0.0", 0.0, 8.0, "c", "m")
    sc.flush()
    assert (out / "captions.csv").exists()
    assert not (tmp_path / "captions.csv").exists()


def test_check_writable_fails_before_spending(tmp_path, monkeypatch):
    """Tests run as root in the container, where a 0o500 dir is no barrier —
    so raise the OSError a read-only mount would raise."""
    sc = CaptionSidecar(tmp_path / "a.mp4")

    def read_only(*a, **k):
        raise OSError("Read-only file system")

    monkeypatch.setattr(Path, "mkdir", read_only)
    with pytest.raises(RuntimeError, match="read-only"):
        sc.check_writable()


# --- the indexer path -----------------------------------------------------

def test_indexer_appends_caption_to_the_transcript_text(tmp_path, monkeypatch):
    from moment_miner import indexer

    monkeypatch.setattr(indexer, "probe", lambda p: {"duration": 8.0, "has_audio": False})
    monkeypatch.setattr(indexer, "extract_windows",
                        lambda path, spans, **k: ((t0, t1, frames(8)) for t0, t1 in spans))

    class M:
        def pending(self, settings=None): return [{"id": "a", "path": str(tmp_path / "a.mp4")}]
        def transcript_between(self, *a): return "spoken words"
        def mark_done(self, *a): pass
        def mark_error(self, *a): raise AssertionError("indexing failed")

    class S:
        def __init__(self): self.rows = []
        def delete_path(self, p): pass
        def add(self, rows): self.rows += rows
        def rebuild_fts(self): pass

    class E:
        name = "mock"
        def embed_segment(self, f): return np.zeros(4, dtype=np.float32)

    store = S()
    counts = indexer.index_pending(
        M(), store, E(), use_asr=False,
        caption_backend=MockCaptionBackend(), log=lambda *a: None,
    )
    assert counts["captioned"] == len(store.rows)
    assert all(r["text"].startswith("spoken words mock caption") for r in store.rows)


def test_indexer_without_caption_backend_spends_nothing(tmp_path, monkeypatch):
    from moment_miner import indexer

    monkeypatch.setattr(indexer, "probe", lambda p: {"duration": 8.0, "has_audio": False})
    monkeypatch.setattr(indexer, "extract_windows",
                        lambda path, spans, **k: ((t0, t1, frames(8)) for t0, t1 in spans))

    class Boom(MockCaptionBackend):
        def caption(self, *a): raise AssertionError("captioned without being asked")

    class M:
        def pending(self, settings=None): return [{"id": "a", "path": str(tmp_path / "a.mp4")}]
        def transcript_between(self, *a): return "spoken words"
        def mark_done(self, *a): pass
        def mark_error(self, *a): raise AssertionError("indexing failed")

    class S:
        def __init__(self): self.rows = []
        def delete_path(self, p): pass
        def add(self, rows): self.rows += rows
        def rebuild_fts(self): pass

    class E:
        name = "mock"
        def embed_segment(self, f): return np.zeros(4, dtype=np.float32)

    store = S()
    counts = indexer.index_pending(M(), store, E(), use_asr=False, log=lambda *a: None)
    assert "captioned" not in counts
    assert all(r["text"] == "spoken words" for r in store.rows)
    assert not (tmp_path / "captions.csv").exists()


# --- the standalone pass over an existing index ---------------------------

def test_still_times_are_the_instants_subsample_picks():
    for span, n in ((8.0, None), (3.0, None), (6.0, None), (8.0, 5), (8.0, 1)):
        picked = subsample(np.arange(8), span, 4.0, n)
        times = still_times(10.0, 10.0 + span, 4.0, n, frames_per_window=8)
        assert times == [10.0 + i * span / 8 for i in picked]


class Recording(MockCaptionBackend):
    name = "recording"

    def __init__(self, text="kong vault over a rail"):
        self.text, self.shapes = text, []

    def caption(self, frames, span, n_frames=None, stride=4.0):
        self.shapes.append(frames.shape)
        return self.text


class Refuses(MockCaptionBackend):
    def caption(self, *a, **k):
        raise AssertionError("bought a caption the sidecar already had")


def _indexed(tmp_path, synthetic_video):
    from moment_miner.embeddings.mock import MockBackend
    from moment_miner.indexer import index_pending
    from moment_miner.manifest import Manifest
    from moment_miner.store import SegmentStore

    manifest = Manifest(tmp_path / "data" / "manifest.db")
    manifest.scan(synthetic_video.parent)
    store = SegmentStore(tmp_path / "data", "mock")
    index_pending(manifest, store, MockBackend(), use_asr=False, log=lambda *a: None)
    return manifest, store


@requires_ffmpeg
def test_caption_pass_over_an_existing_index(tmp_path, synthetic_video):
    from moment_miner.indexer import caption_indexed

    manifest, store = _indexed(tmp_path, synthetic_video)
    before = {r["id"]: list(map(float, r["vector"]))
              for r in store.table.search().limit(100).to_list()}
    assert store.text_search("kong") == []

    be = Recording()
    counts = caption_indexed(manifest, store, be, synthetic_video.parent,
                             frame_w=200, frame_h=120, log=lambda *a: None)

    assert counts["captioned"] == len(before) == 2
    rows = store.table.search().limit(100).to_list()
    assert all(r["text"] == "kong vault over a rail" for r in rows)
    assert {r["id"]: list(map(float, r["vector"])) for r in rows} == before
    assert {h["id"] for h in store.text_search("kong")} == set(before)
    sidecar = CaptionSidecar(synthetic_video)
    assert all(sidecar.get(i) == "kong vault over a rail" for i in before)


@requires_ffmpeg
def test_caption_pass_decodes_at_its_own_raster(tmp_path, synthetic_video):
    from moment_miner.frames import FRAME_H, FRAME_W
    from moment_miner.indexer import caption_indexed

    manifest, store = _indexed(tmp_path, synthetic_video)
    be = Recording()
    caption_indexed(manifest, store, be, synthetic_video.parent,
                    frame_w=200, frame_h=120, log=lambda *a: None)
    assert (120, 200) != (FRAME_H, FRAME_W)
    # 8 s windows at stride 4 take two stills each, at 2 s and 6 s.
    assert be.shapes == [(2, 120, 200, 3)] * 2


@requires_ffmpeg
def test_caption_pass_skips_segments_the_sidecar_has(tmp_path, synthetic_video):
    from moment_miner.indexer import caption_indexed

    manifest, store = _indexed(tmp_path, synthetic_video)
    caption_indexed(manifest, store, Recording(), synthetic_video.parent,
                    log=lambda *a: None)
    counts = caption_indexed(manifest, store, Refuses(), synthetic_video.parent,
                             log=lambda *a: None)
    assert counts["captioned"] == 0 and counts["reused"] == 2

    forced = Recording("backflip off a wall")
    counts = caption_indexed(manifest, store, forced, synthetic_video.parent,
                             force=True, log=lambda *a: None)
    assert counts["captioned"] == 2 and len(forced.shapes) == 2
    assert len(store.text_search("backflip")) == 2
    assert store.text_search("kong") == []


@requires_ffmpeg
def test_mm_caption_cli(tmp_path, synthetic_video):
    from click.testing import CliRunner

    from moment_miner.cli import main

    _indexed(tmp_path, synthetic_video)
    out = tmp_path / "captions-out"
    result = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "data"), "caption", str(synthetic_video.parent),
        "--model", "mock", "--backend", "mock", "--frame-size", "200x120",
        "--caption-dir", str(out), "--caption-frames", "3",
    ])
    assert result.exit_code == 0, result.output
    assert "'captioned': 2" in result.output
    rows = list(csv.DictReader((out / "captions.csv").open()))
    assert [r["caption"] for r in rows] == ["mock caption of 3 frames over 8.0s"] * 2

    bad = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "data"), "caption", str(synthetic_video.parent),
        "--model", "mock", "--backend", "mock", "--frame-size", "wide"])
    assert bad.exit_code != 0 and "WxH" in bad.output


def test_shipped_prompts_load_and_differ():
    general = load_prompt("general")
    parkour = load_prompt("parkour")
    assert "{n}" in general and "{span:.0f}" in general
    assert "parkour" in parkour.lower() and "parkour" not in general.lower()
    assert {"general", "parkour"} <= set(prompt_names())


def test_prompt_can_come_from_a_file(tmp_path):
    p = tmp_path / "mine.txt"
    p.write_text("{n} frames, {span:.0f}s")
    assert load_prompt(str(p)) == "{n} frames, {span:.0f}s"


def test_unknown_prompt_names_the_shipped_ones():
    with pytest.raises(ValueError, match="general"):
        load_prompt("nope")
