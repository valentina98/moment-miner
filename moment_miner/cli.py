import csv
import json as jsonlib
import os
from importlib.resources import files as pkg_files
from pathlib import Path

import click

from .manifest import Manifest
from .store import SegmentStore
from .timefmt import fmt_ts, parse_ts


@click.group()
@click.version_option(package_name="moment-miner", prog_name="mm")
@click.option("--data-dir", default="mm_data", show_default=True,
              help="Where the index lives.")
@click.pass_context
def main(ctx, data_dir):
    """Moment Miner: search large video archives, export lossless clips."""
    ctx.obj = Path(data_dir)


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
@click.pass_obj
def index(data_dir, folder, backend, window, stride, asr, asr_model, reindex):
    """Scan FOLDER recursively and index new/changed videos."""
    from .embeddings import get_backend
    from .indexer import index_pending

    manifest = Manifest(data_dir / "manifest.db")
    counts = manifest.scan(folder)
    if reindex:
        manifest.reset(folder)
    click.echo(f"scan: {counts}")
    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    result = index_pending(
        manifest, store, be, use_asr=asr, asr_model=asr_model,
        win=window, stride=stride, log=click.echo,
    )
    click.echo(f"done: {result}")


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
@click.pass_obj
def search(data_dir, query, k, backend, as_json, llc, llc_dir):
    """Semantic + transcript search; prints ranked timestamps."""
    from .embeddings import get_backend
    from .search import search as run_search

    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    hits = run_search(query, be, store, k=k)
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
    from .indexer import index_pending
    from .search import search as run_search

    manifest = Manifest(data_dir / "manifest.db")
    manifest.scan(folder)
    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    pending = manifest.pending()
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
            csv.writer(f).writerow([query, videos[idx - 1].name, start, end])
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
