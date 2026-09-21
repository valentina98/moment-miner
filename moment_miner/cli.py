import csv
import json as jsonlib
import os
from importlib.resources import files as pkg_files
from pathlib import Path

import click

from .locate import LOCATORS
from .manifest import Manifest
from .motion import STATIC_MAX
from .store import MotionStore, SegmentStore
from .timefmt import fmt_ts, parse_ts


@click.group()
@click.version_option(package_name="moment-miner", prog_name="mm")
@click.option("--data-dir", default="mm_data", show_default=True,
              help="Where the index lives.")
@click.pass_context
def main(ctx, data_dir):
    """Moment Miner: search large video archives, export lossless clips."""
    ctx.obj = Path(data_dir)


def _prompt_names() -> list[str]:
    from .captions import prompt_names

    return prompt_names()


@main.command(name="help")
@click.argument("command", required=False)
@click.pass_context
def help_cmd(ctx, command):
    """Show help for mm or a specific COMMAND."""
    if command is None:
        click.echo(main.get_help(ctx.parent))
        return
    cmd = main.get_command(ctx, command)
    if cmd is None:
        raise click.BadParameter(
            f"unknown command {command!r}; commands: "
            + ", ".join(main.list_commands(ctx)))
    click.echo(cmd.get_help(click.Context(cmd, info_name=f"mm {command}")))


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--backend", default="siglip", show_default=True)
@click.option("--window", default=8.0, show_default=True, help="Segment length (s).")
@click.option("--stride", default=4.0, show_default=True, help="Segment stride (s).")
@click.option("--asr/--no-asr", default=True, show_default=True)
@click.option("--asr-model", default="small", show_default=True)
@click.option("--reindex", is_flag=True,
              help="Re-process videos even if already indexed.")
@click.option("--caption", default=None,
              help="Caption each segment with a Claude model (e.g. "
                   "claude-haiku-4-5) and add it to the searchable text. "
                   "Spends session quota via Claude Code credentials; "
                   "ANTHROPIC_API_KEY spends money and needs --allow-paid "
                   "with --paid-budget-usd. Captions are cached beside the "
                   "footage.")
@click.option("--allow-paid", is_flag=True, default=False,
              help="Permit the ANTHROPIC_API_KEY route, which spends money. "
                   "Refused without --paid-budget-usd.")
@click.option("--paid-budget-usd", type=float, default=None,
              help="The cap you accept for this pass, in USD. Required by "
                   "--allow-paid; a gate before the first request, not a meter.")
@click.option("--caption-dir", default=None,
              help="Write caption sidecars here instead of beside each video "
                   "(use when the archive is mounted read-only).")
@click.option("--caption-prompt", default="general", show_default=True,
              help="Caption prompt: a shipped name (see `mm caption --help`) or a file path.")
@click.option("--caption-frames", default=None, type=int,
              help="Force the stills sent per segment. By default the count "
                   "follows from the segment's span and --stride. Each still "
                   "is ~170 input tokens, the dominant cost of a caption pass.")
@click.option("--caption-frames-short", default=None, type=int,
              help="Force the count for short videos instead.")
@click.option("--caption-frames-long", default=None, type=int,
              help="Stills per segment for long videos instead. Defaults to "
                   "--caption-frames.")
@click.option("--short-video", "short_video_s", default=8.0, show_default=True,
              help="A video no longer than this many seconds counts as short. "
                   "The default is one window.")
@click.option("--long-video", "long_video_s", default=120.0, show_default=True,
              help="A video at least this many seconds counts as long.")
@click.pass_obj
def index(data_dir, folder, backend, window, stride, asr, asr_model, reindex,
          caption, allow_paid, paid_budget_usd, caption_dir, caption_prompt,
          caption_frames,
          caption_frames_short, caption_frames_long, short_video_s,
          long_video_s):
    """Scan FOLDER recursively and index new/changed videos."""
    from .captions import MissingCaptionCredentials, get_caption_backend
    from .embeddings import get_backend
    from .indexer import index_pending

    manifest = Manifest(data_dir / "manifest.db")
    counts = manifest.scan(folder)
    if reindex:
        manifest.reset(folder)
    click.echo(f"scan: {counts}")
    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    try:
        result = index_pending(
            manifest, store, be, use_asr=asr, asr_model=asr_model,
            motion_store=MotionStore(data_dir),
            win=window, stride=stride, log=click.echo,
            caption_backend=get_caption_backend(
                caption, allow_paid=allow_paid, prompt=caption_prompt,
                paid_budget_usd=paid_budget_usd) if caption else None,
            caption_dir=caption_dir,
            caption_frames=caption_frames,
            caption_frames_short=caption_frames_short,
            caption_frames_long=caption_frames_long,
            short_video_s=short_video_s,
            long_video_s=long_video_s,
        )
    except MissingCaptionCredentials as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"done: {result}")


def _frame_size(ctx, param, value):
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise click.BadParameter(f"expected WxH, e.g. 640x360; got {value!r}")
    if w <= 0 or h <= 0:
        raise click.BadParameter(f"both sides must be positive; got {value!r}")
    return w, h


@main.command(
    epilog="Caption prompts: " + ", ".join(_prompt_names()))
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--model", default="claude-haiku-4-5", show_default=True,
              help="Claude model id, or 'mock'. Spends session quota via "
                   "Claude Code credentials; ANTHROPIC_API_KEY is not used "
                   "here.")
@click.option("--backend", default="siglip", show_default=True,
              help="Which index to caption. The embedding model is not loaded.")
@click.option("--frame-size", default="640x360", show_default=True,
              callback=_frame_size,
              help="Raster the stills are decoded at, independent of the "
                   "embedding's. Stills wider than 640 px are downscaled "
                   "before sending.")
@click.option("--caption-dir", default=None,
              help="Write caption sidecars here instead of beside each video.")
@click.option("--caption-prompt", default="general", show_default=True,
              help="Caption prompt: a shipped name (see `mm caption --help`) or a file path.")
@click.option("--caption-frames", default=None, type=int,
              help="Force the stills sent per segment. By default the count "
                   "follows from the segment's span and the indexed stride.")
@click.option("--caption-frames-short", default=None, type=int,
              help="Force the count for short videos instead.")
@click.option("--caption-frames-long", default=None, type=int,
              help="Force the count for long videos instead.")
@click.option("--short-video", "short_video_s", default=8.0, show_default=True)
@click.option("--long-video", "long_video_s", default=120.0, show_default=True)
@click.option("--force", is_flag=True,
              help="Caption again even where the sidecar already has one.")
@click.pass_obj
def caption(data_dir, folder, model, backend, frame_size, caption_dir,
            caption_prompt, caption_frames, caption_frames_short, caption_frames_long,
            short_video_s, long_video_s, force):
    """Caption the already-indexed segments under FOLDER.

    Reads the segments from the index instead of re-embedding, decodes only
    the stills each caption needs, and rewrites the searchable text.
    """
    from .captions import get_caption_backend
    from .indexer import caption_indexed

    store = SegmentStore(data_dir, backend)
    try:
        result = caption_indexed(
            Manifest(data_dir / "manifest.db"), store,
            get_caption_backend(model, prompt=caption_prompt),
            folder, frame_w=frame_size[0], frame_h=frame_size[1],
            caption_dir=caption_dir, caption_frames=caption_frames,
            caption_frames_short=caption_frames_short,
            caption_frames_long=caption_frames_long,
            short_video_s=short_video_s, long_video_s=long_video_s,
            force=force, log=click.echo,
        )
    except RuntimeError as e:
        # Missing credentials, a read-only archive, or no index yet.
        raise click.ClickException(str(e)) from e
    click.echo(f"done: {result}")


@main.command("caption-compare")
@click.argument("caption_files", nargs=-1, required=True,
                type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--out", default=None, type=click.Path(dir_okay=False),
              help="Where to write the page. Defaults to "
                   "caption-comparison.html inside --videos, so everything "
                   "about an archive stays on the archive.")
@click.option("--videos", default=None, type=click.Path(exists=True, file_okay=False),
              help="Folder holding the videos, so the page can show each "
                   "segment's stills. Omit for a text-only page.")
@click.option("--segments", "segments_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="seg,video,t0,t1 CSV, needed when the caption files carry "
                   "only seg,caption instead of the sidecar columns.")
@click.option("--frames-dir", default=None, type=click.Path(exists=True, file_okay=False),
              help="Use stills already extracted here, named <seg id>_*.jpg, "
                   "instead of decoding the videos.")
@click.option("--blind/--no-blind", default=True, show_default=True,
              help="Hide which file wrote which caption and shuffle their "
                   "order, so a comparison can be judged without knowing.")
@click.option("--seed", default=0, show_default=True, help="Shuffle seed.")
def caption_compare(caption_files, out, videos, segments_file, frames_dir, blind, seed):
    """Render CAPTION_FILES side by side as one HTML page.

    Marks where the models disagree, and by default hides which file is which
    so the result can be judged blind.
    """
    from .compare import build

    html, order_map = build(
        [Path(f) for f in caption_files], videos=Path(videos) if videos else None,
        segments_file=Path(segments_file) if segments_file else None,
        frames_dir=Path(frames_dir) if frames_dir else None,
        blind=blind, seed=seed,
    )
    out_path = Path(out) if out else (
        Path(videos) if videos else Path(".")) / "caption-comparison.html"
    out_path.write_text(html)
    click.echo(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    if blind:
        key_path = out_path.with_name(out_path.stem + "-key.csv")
        with key_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(order_map[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(order_map)
        click.echo(f"key: {key_path} — leave it shut until every segment is judged")


@main.command()
@click.argument("query")
@click.option("-k", default=10, show_default=True, help="Max results.")
@click.option("--backend", default="siglip", show_default=True)
@click.option("--json", "as_json", is_flag=True)
@click.option("--llc", is_flag=True,
              help="Write LosslessCut projects (.llc) into an llc/ subfolder "
                   "next to each matched video.")
@click.option("--llc-dir", default=None,
              help="Write .llc files here instead; media is referenced "
                   "relatively, so open them in place (don't move them).")
@click.option("--static/--moving", "static", default=None,
              help="Keep only locked-off shots, or only shots where most of "
                   "the frame changes. A filter, not a search term. Segments "
                   "with no motion value carry none and drop out.")
@click.option("--static-max", default=STATIC_MAX, show_default=True,
              help="A shot is static when fewer than this fraction of pixels "
                   "change between frames. Applied here, not at index time, "
                   "so retuning it costs nothing.")
@click.pass_obj
def search(data_dir, query, k, backend, as_json, llc, llc_dir, static,
           static_max):
    """Semantic + transcript search; prints ranked timestamps."""
    from .embeddings import get_backend
    from .search import search as run_search

    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    hits = run_search(query, be, store, k=k, static=static,
                      motion_store=MotionStore(data_dir),
                      static_max=static_max)
    if (llc or llc_dir) and hits:
        from .export import write_llc_projects

        for f in write_llc_projects(hits, llc_dir, label=query):
            click.echo(f"llc: {f}")
    if as_json:
        click.echo(jsonlib.dumps(
            [{k_: h[k_] for k_ in ("path", "t0", "t1", "score", "text")}
             for h in hits], indent=2))
        return
    if not hits:
        click.echo("no results")
        return
    for h in hits:
        click.echo(f"{h['path']}\n  {fmt_ts(h['t0'])} - {fmt_ts(h['t1'])}"
                   f"  (score {h['score']:.4f})")
        if h.get("text"):
            click.echo(f"  \"{h['text'][:120]}\"")


@main.command()
@click.argument("video")
@click.argument("t0")
@click.argument("t1")
@click.option("-o", "--output", default=None, help="Output file path.")
@click.option("--pad", default=1.0, show_default=True)
@click.option("--snap/--no-snap", default=True, show_default=True,
              help="Snap start to previous keyframe (fully lossless).")
@click.option("--smart/--no-smart", default=True, show_default=True,
              help="Frame-accurate cut (recodes only the boundary GOPs). "
                   "--no-smart is a pure stream copy, snapped to a keyframe.")
@click.pass_obj
def export(data_dir, video, t0, t1, output, pad, snap, smart):
    """Export [T0, T1] of VIDEO (path or index id) as a lossless clip."""
    from .export import export_clip

    manifest = Manifest(data_dir / "manifest.db")
    row = manifest.video(video)
    path = row["path"] if row else video
    if not Path(path).is_file():
        raise click.BadParameter(f"video not found: {video}")
    start, end = parse_ts(t0), parse_ts(t1)
    if output is None:
        stem = Path(path).stem
        output = f"{stem}_{fmt_ts(start).replace(':', '-')}_{fmt_ts(end).replace(':', '-')}.mp4"
    out, a0, a1 = export_clip(path, start, end, output, pad=pad, snap=snap, smart=smart)
    click.echo(f"wrote {out} ({fmt_ts(a0)} - {fmt_ts(a1)} of source)")


@main.command()
@click.argument("labels", type=click.Path(exists=True, dir_okay=False))
@click.option("-k", default=10, show_default=True)
@click.option("--backend", default="siglip", show_default=True)
@click.pass_obj
def eval(data_dir, labels, k, backend):
    """Score search quality against a labels CSV (query,video,start,end)."""
    from .embeddings import get_backend
    from .evaluate import evaluate, load_labels

    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    report = evaluate(load_labels(labels), be, store, k=k)
    for q in report["per_query"]:
        click.echo(f"  {q['found']}/{q['total']}  {q['query']}")
    click.echo(
        f"recall@{report['k']}: {report['recall']:.2f}"
        f"   MRR: {report['mrr']:.2f}   ({report['n_labels']} labels)"
    )


def _default_mine_out(folder: str, query: str) -> Path:
    import re
    from datetime import datetime

    src = Path(folder).resolve()
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:40] or "query"
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return src.parent / f"{src.name}_mined" / f"{slug}_{stamp}"


@main.command()
@click.argument("query")
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("-o", "--output", default=None,
              help="Folder for the exported clips "
                   "[default: <FOLDER>_mined/<query>_<timestamp>/ next to FOLDER].")
@click.option("-k", default=5, show_default=True, help="Export top-k matches.")
@click.option("--backend", default="siglip", show_default=True)
@click.option("--pad", default=1.0, show_default=True)
@click.option("--asr/--no-asr", default=True, show_default=True)
@click.option("--smart/--no-smart", default=True, show_default=True,
              help="Frame-accurate cuts; --no-smart is faster but snaps to keyframes.")
@click.pass_obj
def mine(data_dir, query, folder, output, k, backend, pad, asr, smart):
    """One shot: index FOLDER (only new/changed videos), search QUERY,
    export the top-k matching moments as lossless clips into OUTPUT."""
    from .embeddings import get_backend
    from .export import export_clip
    from .indexer import index_pending, index_settings
    from .search import search as run_search

    manifest = Manifest(data_dir / "manifest.db")
    manifest.scan(folder)
    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    pending = manifest.pending(index_settings(be.name))
    if pending:
        click.echo(f"indexing {len(pending)} new/changed video(s)…")
        index_pending(manifest, store, be, use_asr=asr, log=click.echo)
    prefix = str(Path(folder).resolve()) + os.sep
    hits = [h for h in run_search(query, be, store, k=50)
            if h["path"].startswith(prefix)][:k]
    if not hits:
        click.echo("no results")
        return
    output = Path(output) if output else _default_mine_out(folder, query)
    output.mkdir(parents=True, exist_ok=True)
    for i, h in enumerate(hits, 1):
        name = f"{i:02d}_{Path(h['path']).stem}_{fmt_ts(h['t0']).replace(':', '-')}.mp4"
        out, _, _ = export_clip(h["path"], h["t0"], h["t1"],
                                str(Path(output) / name), pad=pad, smart=smart)
        click.echo(f"{out}  ← {h['path']} {fmt_ts(h['t0'])}-{fmt_ts(h['t1'])}")


def _ts_or_reprompt(value: str) -> float:
    try:
        return parse_ts(value)
    except ValueError as e:
        raise click.UsageError(str(e))


def _template_names() -> list[str]:
    tdir = pkg_files("moment_miner") / "templates"
    return sorted(p.name[len("labels_"):-len(".csv")]
                  for p in tdir.iterdir() if p.name.endswith(".csv"))


@main.command(
    epilog="Available templates: " + ", ".join(_template_names()))
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--template", "template_name", default=None,
              help="Start labels.csv from a template if it doesn't exist yet.")
@click.option("--labels", "labels_path", default=None,
              help="Labels file [default: FOLDER/labels.csv].")
def annotate(folder, template_name, labels_path):
    """Interactively append ground-truth labels for videos in FOLDER.

    Labels live with the footage (FOLDER/labels.csv) so `mm eval` and
    `make eval` find them. Watch the video in any player, then enter the
    times here.
    """
    from .manifest import VIDEO_EXTS

    labels = Path(labels_path) if labels_path else Path(folder) / "labels.csv"
    if not labels.exists():
        if template_name:
            src = pkg_files("moment_miner") / "templates" / f"labels_{template_name}.csv"
            if not src.is_file():
                raise click.BadParameter(
                    f"unknown template {template_name!r}; available: {_template_names()}")
            labels.write_text(src.read_text())
            click.echo(f"created {labels} from template {template_name!r}")
        else:
            labels.write_text("query,video,start,end\n")
            click.echo(f"created empty {labels}"
                       f" (templates available: {_template_names()})")
    videos = sorted(p for p in Path(folder).rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise click.ClickException(f"no videos found under {folder}")
    click.echo("videos:")
    for i, v in enumerate(videos, 1):
        click.echo(f"  [{i}] {v.name}")
    # Append against the file's own header, not a fixed field order: a labels file may carry
    # extra columns (the parkour template has `category`) and a positional write would shift
    # every field one place.
    with labels.open(newline="") as f:
        fieldnames = next(csv.reader(f))
    added = 0
    while True:
        query = click.prompt("query (empty to finish)", default="",
                             show_default=False).strip()
        if not query:
            break
        idx = click.prompt(f"video [1-{len(videos)}]", type=click.IntRange(1, len(videos)))
        start = click.prompt("start (M:SS or seconds)", value_proc=_ts_or_reprompt)
        end = click.prompt("end   (M:SS or seconds)", value_proc=_ts_or_reprompt)
        if end <= start:
            click.echo("end must be after start — row skipped")
            continue
        with labels.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames).writerow(
                {"query": query, "video": videos[idx - 1].name,
                 "start": start, "end": end})
        added += 1
        click.echo(f"added ({added} this session)")
    click.echo(f"done — {added} label(s) appended to {labels}")
    if added:
        click.echo(f"score the index anytime:  mm eval {labels}")


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7700, show_default=True)
@click.option("--backend", default="siglip", show_default=True)
@click.pass_obj
def serve(data_dir, host, port, backend):
    """Search daemon: models stay resident, queries are sub-second warm.

    GET /health, GET /search?q=...&k=10 (JSON).
    """
    from .embeddings import get_backend
    from .serve import make_server

    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    be.embed_text(["warmup"])
    server = make_server(be, store, host, port)
    click.echo(f"listening on http://{host}:{port}"
               f" (backend {be.name}, {store.count()} segments)")
    server.serve_forever()


@main.command()
@click.option("--errors", "show_errors", is_flag=True,
              help="List files that failed to index, with the error.")
@click.pass_obj
def status(data_dir, show_errors):
    """Index statistics."""
    manifest = Manifest(data_dir / "manifest.db")
    for row in manifest.stats():
        click.echo(f"{row['status']:>8}: {row['n']} videos, {row['dur'] / 3600:.1f} h")
    if show_errors:
        for row in manifest.errors():
            click.echo(f"  {row['path']}: {row['error']}")


if __name__ == "__main__":
    main()


def _score_video(path: str, query_vec, backend, fps: float):
    import numpy as np

    from .frames import frame_batches
    from .locate import heatmap

    chunks, stamps = [], []
    for frames, ts in frame_batches(path, fps):
        chunks.append(heatmap(backend.embed_frames(frames), query_vec))
        stamps.append(ts)
    if not chunks:
        return np.array([]), np.array([])
    return np.concatenate(chunks), np.concatenate(stamps)


@main.command()
@click.option("-q", "--query", help="One query to locate.")
@click.option("-v", "--video", type=click.Path(exists=True, dir_okay=False),
              help="One video to search inside.")
@click.option("--labels", type=click.Path(exists=True, dir_okay=False),
              help="CSV of query,video,start,end — locate every row and shade the label.")
@click.option("--videos", type=click.Path(exists=True, file_okay=False),
              help="Where the videos named by --labels live.")
@click.option("--fps", default=5.0, show_default=True,
              help="Frames scored per second. The resolution knob, set per query.")
@click.option("--locator", "locator_name", default="maxsub", show_default=True,
              type=click.Choice(sorted(LOCATORS)))
@click.option("--duration", type=float,
              help="Region length in seconds. Required by --locator window, refused by maxsub.")
@click.option("--top", "top_k", default=1, show_default=True, help="Regions per video.")
@click.option("--backend", default="siglip", show_default=True)
@click.option("-o", "--out", type=click.Path(dir_okay=False),
              help="Write an HTML page of the score curves.")
def locate(query, video, labels, videos, fps, locator_name, duration, top_k, backend, out):
    """Score a video frame by frame against a query and pick the region to cut.

    The index ranks whole segments; this says where inside one the action is.
    """
    from .embeddings import get_backend
    from .locate import build_locator

    if labels:
        if not videos:
            raise click.UsageError("--labels needs --videos")
        rows = [r for r in csv.DictReader(open(labels, newline=""))
                if r.get("video", "").strip()]
    elif query and video:
        rows = [{"query": query, "video": video, "start": "", "end": ""}]
    else:
        raise click.UsageError("give --query and --video, or --labels and --videos")

    be = get_backend(backend)
    loc = build_locator(locator_name, duration=duration, top_k=top_k)
    root = Path(videos) if videos else None
    panels = []
    for r in rows:
        path = r["video"] if root is None else next(
            (str(p) for p in root.rglob(r["video"])), None)
        if path is None:
            click.echo(f"  skipped {r['video']}: not found under {videos}")
            continue
        qv = be.embed_text([r["query"]])[0]
        scores, times = _score_video(path, qv, be, fps)
        if scores.size == 0:
            click.echo(f"  skipped {r['video']}: no frames decoded")
            continue
        regions = loc.locate(scores, times)
        click.echo(f"{r['query']}  [{Path(path).name}]")
        for g in regions:
            click.echo(f"  {fmt_ts(g.t0)} - {fmt_ts(g.t1)}"
                       f"  ({g.t1 - g.t0:.1f}s, peak {g.peak:.4f})")
        panels.append({
            "query": r["query"], "video": Path(path).name,
            "times": [round(float(t), 3) for t in times],
            "scores": [round(float(s), 5) for s in scores],
            "regions": [{"t0": g.t0, "t1": g.t1, "peak": g.peak} for g in regions],
            "label": ([float(r["start"]), float(r["end"])]
                      if r.get("start", "").strip() and r.get("end", "").strip() else None),
        })

    if out:
        template = (pkg_files("moment_miner") / "templates" / "locate.html").read_text()
        Path(out).write_text(template.replace(
            "__PANELS__", jsonlib.dumps(panels)).replace(
            "__META__", jsonlib.dumps({"fps": fps, "locator": locator_name,
                                       "duration": duration, "backend": be.name})))
        click.echo(f"wrote {out} ({len(panels)} panel(s))")
