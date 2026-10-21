import csv
from collections import Counter
import json as jsonlib
import re
import time
from importlib.resources import files as pkg_files
from pathlib import Path

import click

from . import guard
from .locate import LOCATORS
from .hostpaths import UserPath, on_host
from .manifest import Manifest
from .motion import STATIC_MAX
from .store import AxisStore, SegmentStore
from .timefmt import fmt_ts, parse_ts


@click.group()
@click.version_option(package_name="moment-miner", prog_name="mm")
@click.option("--data-dir", default="~/.mm_data", show_default=True, type=UserPath(file_okay=False),
              help="Where everything the tool keeps lives: the index, the "
                   "caption cache, labels and exported clips (cuts/). Any "
                   "folder works, a footage folder included.")
@click.pass_context
def main(ctx, data_dir):
    """Moment Miner: search large video archives, cut lossless clips."""
    ctx.obj = Path(data_dir).expanduser()
    guard.arm(ctx.obj)
    ctx.call_on_close(guard.disarm)


def _ranker_names() -> list[str]:
    from .rankers import ranker_names

    return ranker_names()


def _set_name(ctx, param, value):
    """A caption set's name, also used as a file name by `caption-export`."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise click.BadParameter(
            f"{value!r}: use letters, digits, '.', '_' and '-', starting with a letter or digit")
    return value


def _set_names(ctx, param, values):
    return tuple(_set_name(ctx, param, v) for v in values)


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
@click.argument("folder", type=UserPath(exists=True, file_okay=False))
@click.option("--backend", default="siglip", show_default=True)
@click.option("--window", default=8.0, show_default=True, help="Segment length (s).")
@click.option("--stride", default=4.0, show_default=True, help="Segment stride (s).")
@click.option("--asr/--no-asr", default=True, show_default=True)
@click.option("--asr-model", default="small", show_default=True)
@click.option("--reindex", is_flag=True,
              help="Re-process videos even if already indexed.")
@click.option("--estimate/--no-estimate", default=True, show_default=True,
              help="Print an ETA first. The first run on a machine measures its "
                   "speed once on 24 s of the first video (`mm calibrate`).")
@click.pass_obj
def index(data_dir, folder, backend, window, stride, asr, asr_model, reindex, estimate):
    """Scan FOLDER recursively and index new, changed and failed videos.

    Never captions and never spends: `mm caption` does that. A re-indexed
    segment keeps the captions it already had.
    """
    from tqdm import tqdm

    from .embeddings import get_backend
    from .indexer import index_pending, index_settings

    manifest = Manifest(data_dir / "manifest.db")
    counts = manifest.scan(folder)
    if reindex:
        manifest.reset(folder)
    be = get_backend(backend)
    _forget_missing(manifest, counts, data_dir, be.name, folder)
    click.echo(f"scan: {counts}")
    settings = index_settings(be.name, window, stride)
    _report_stale_elsewhere(manifest, settings, folder)
    pending = manifest.pending(settings, folder)
    eta, segments, failed = (_estimate(data_dir, be, pending, window, stride, asr)
                             if estimate and pending else (None, None, {}))
    store = SegmentStore(data_dir, be.name)
    started = time.time()
    # tqdm, not click.progressbar: click draws nothing when stdout is not a
    # terminal, and `make index` runs docker without -t. Log lines still go
    # through click.echo, which flushes; tqdm.write does not, and a piped
    # stdout would hold every line back until the run ends.
    def log(msg):
        with tqdm.external_write_mode():
            click.echo(msg)

    with tqdm(total=segments, unit="segment", disable=not pending, mininterval=1.0,
              dynamic_ncols=True) as bar:
        result = index_pending(
            manifest, store, be, use_asr=asr, asr_model=asr_model,
            axis_store=AxisStore(data_dir),
            win=window, stride=stride, log=log, progress=bar.update,
            folder=folder,
            failed=failed,
        )
    took = f"; took {(time.time() - started) / 60:.1f} min"
    if eta is not None:
        took += f", estimated {eta / 60:.1f}"
    click.echo(f"done: {result}{took if pending else ''}")


def _estimate(data_dir, be, pending, window, stride, asr):
    """The ETA line, the segment count the progress bar fills, and the files
    whose probe failed here, which the index pass marks failed without waiting
    on the same hung probe a second time."""
    from .calibrate import calibrate, estimate, load_rate

    rate = load_rate(data_dir, be.name)
    try:
        if rate is None:
            click.echo("calibrating: first index on this machine, measuring its "
                       "speed once (about a minute)...")
            rate = calibrate(data_dir, be, pending[0]["path"], log=click.echo)
        segments, seconds, eta, failed = estimate(pending, rate, window, stride)
    except Exception as e:
        click.echo(f"estimate: unavailable ({e})")
        return None, None, {}
    for path, error in failed.items():
        click.echo(f"estimate: {on_host(path)} could not be probed, skipped this run ({error})")
    note = "; speech recognition comes on top" if asr else ""
    click.echo(f"estimate: {len(pending) - len(failed)} video(s), {seconds / 60:.1f} min of "
               f"video, {segments} segments, ~{eta / 60:.1f} min at "
               f"{rate['s_per_segment']} s/segment{note}")
    return eta, segments, failed


@main.command()
@click.argument("video", type=UserPath(exists=True, dir_okay=False))
@click.option("--backend", default="siglip", show_default=True)
@click.pass_obj
def calibrate(data_dir, video, backend):
    """Measure this machine's index speed on the first seconds of VIDEO.

    Pick a video from the camera you index most. `mm index` runs this by
    itself on the first video it indexes; run it again after a hardware change.
    """
    from .calibrate import calibrate as run
    from .embeddings import get_backend

    run(data_dir, get_backend(backend), video, log=click.echo)


def _forget_missing(manifest, counts, data_dir, backend_name, folder):
    """Drop videos whose file is gone, unless every one under FOLDER is gone.

    All of them missing at once is what a different mount root looks like,
    not a deletion, and forgetting them would throw the whole index away.
    """
    missing = counts.pop("missing")
    counts["removed"] = 0
    if not missing:
        return
    known = counts["unchanged"] + counts["changed"] + len(missing)
    if len(missing) == known:
        click.echo(f"kept: all {len(missing)} indexed video(s) under {on_host(folder)} are "
                   "missing -- a different mount root? Nothing removed.")
        return
    stores = [SegmentStore(data_dir, backend_name), AxisStore(data_dir)]
    for path in missing:
        for store in stores:
            store.delete_path(path)
        manifest.forget(path)
        click.echo(f"removed: {on_host(path)} (file no longer on disk)")
    if stores[0].count():
        stores[0].rebuild_fts()
    counts["removed"] = len(missing)


def _report_stale_elsewhere(manifest, settings, folder):
    inside = {r["path"] for r in manifest.pending(settings, folder)}
    reasons = Counter(manifest.why_pending(r, settings)
                      for r in manifest.pending(settings) if r["path"] not in inside)
    for reason, n in reasons.items():
        click.echo(f"left as is: {n} video(s) outside {on_host(folder)} need re-indexing "
                   f"({reason}); `mm index` on their folder updates them")


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
@click.argument("folder", type=UserPath(exists=True, file_okay=False))
@click.option("--model", default="claude-haiku-4-5", show_default=True,
              help="Claude model id, or 'mock'. Spends session quota via "
                   "Claude Code credentials; ANTHROPIC_API_KEY spends money "
                   "and needs --allow-paid with --paid-budget-usd.")
@click.option("--allow-paid", is_flag=True, default=False,
              help="Permit the ANTHROPIC_API_KEY route, which spends money. "
                   "Refused without --paid-budget-usd.")
@click.option("--paid-budget-usd", type=float, default=None,
              help="The cap you accept for this pass, in USD. Required by "
                   "--allow-paid; a gate before the first request, not a meter.")
@click.option("--backend", default="siglip", show_default=True,
              help="Which index to caption. The embedding model is not loaded.")
@click.option("--frame-size", default="640x360", show_default=True,
              callback=_frame_size,
              help="Raster the stills are decoded at, independent of the "
                   "embedding's. Stills wider than 640 px are downscaled "
                   "before sending.")
@click.option("--caption-set", default="default", show_default=True, callback=_set_name,
              help="Keep this pass's captions under NAME in the cache, e.g. to "
                   "compare two prompts with `mm caption-compare default NAME`.")
@click.option("--caption-prompt", default="general", show_default=True,
              help="Caption prompt: a shipped name (listed below) or a file path.")
@click.option("--caption-frames", default=None, type=int,
              help="Force the stills sent per segment. By default the count "
                   "follows from the segment's span and the indexed stride. "
                   "Each still is ~170 input tokens, the dominant cost of a "
                   "caption pass.")
@click.option("--caption-frames-short", default=None, type=int,
              help="Force the count for short videos instead.")
@click.option("--caption-frames-long", default=None, type=int,
              help="Force the count for long videos instead.")
@click.option("--short-video", "short_video_s", default=8.0, show_default=True,
              help="A video no longer than this many seconds counts as short. "
                   "The default is one window.")
@click.option("--long-video", "long_video_s", default=120.0, show_default=True,
              help="A video at least this many seconds counts as long.")
@click.option("--force", is_flag=True,
              help="Caption again even where the cache already has one.")
@click.pass_obj
def caption(data_dir, folder, model, allow_paid, paid_budget_usd, backend, frame_size,
            caption_set, caption_prompt, caption_frames, caption_frames_short,
            caption_frames_long, short_video_s, long_video_s, force):
    """Caption the already-indexed segments under FOLDER.

    The only command that captions or spends. Reads the segments from the
    index instead of re-embedding, decodes only the stills each caption
    needs, and rewrites the searchable text. Captions are cached in the data
    folder, so a segment is never bought twice.
    """
    from .captions import get_caption_backend
    from .indexer import caption_indexed

    store = SegmentStore(data_dir, backend)
    try:
        result = caption_indexed(
            Manifest(data_dir / "manifest.db"), store,
            get_caption_backend(model, allow_paid=allow_paid, prompt=caption_prompt,
                                paid_budget_usd=paid_budget_usd),
            folder, frame_w=frame_size[0], frame_h=frame_size[1],
            axis_store=AxisStore(data_dir),
            data_dir=data_dir, caption_set=caption_set, caption_frames=caption_frames,
            caption_frames_short=caption_frames_short,
            caption_frames_long=caption_frames_long,
            short_video_s=short_video_s, long_video_s=long_video_s,
            force=force, log=click.echo,
        )
    except RuntimeError as e:
        # Missing credentials, an unwritable cache, or no index yet.
        raise click.ClickException(str(e)) from e
    click.echo(f"done: {result}")


@main.command("caption-compare")
@click.argument("caption_sets", nargs=-1, required=True, callback=_set_names)
@click.option("-o", "--output", default=None, type=UserPath(file_okay=False),
              help="Folder for the page and its key "
                   "[default: DATA_DIR/cuts/caption-compare_<time>/].")
@click.option("--frames-dir", default=None, type=UserPath(exists=True, file_okay=False),
              help="Use stills already extracted here, named <seg id>_*.jpg, "
                   "instead of decoding the videos.")
@click.option("--blind/--no-blind", default=True, show_default=True,
              help="Hide which set wrote which caption and shuffle their "
                   "order, so a comparison can be judged without knowing.")
@click.option("--seed", default=0, show_default=True, help="Shuffle seed.")
@click.pass_obj
def caption_compare(data_dir, caption_sets, output, frames_dir, blind, seed):
    """Render CAPTION_SETS from the cache side by side as one HTML page.

    Compares the segments every set has captioned. Marks where the models
    disagree, and by default hides which set is which so the result can be
    judged blind. Each segment shows its stills when its video is on disk.
    """
    from .captions import CaptionCache
    from .compare import build

    if output:
        guard.output(output)
    sets = {name: CaptionCache(data_dir, name).rows() for name in caption_sets}
    for name, rows in sets.items():
        if not rows:
            raise click.ClickException(f"no captions in set {name!r}; `mm status` lists the sets")
    manifest = Manifest(data_dir / "manifest.db")
    videos = {vid: manifest.video(vid) for vid in {r["video_id"] for rows in sets.values() for r in rows}}
    folder = Path(output) if output else _cut_dir(data_dir, "caption-compare")
    page = _new_file(folder, "caption-comparison.html")
    key_path = _new_file(folder, "caption-comparison-key.csv") if blind else None
    try:
        html, order_map = build(
            sets, videos={vid: row["path"] for vid, row in videos.items() if row},
            frames_dir=Path(frames_dir) if frames_dir else None, blind=blind, seed=seed)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    page.write_text(html)
    click.echo(f"wrote {on_host(page)} ({page.stat().st_size / 1e6:.1f} MB)")
    if key_path:
        with key_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(order_map[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(order_map)
        click.echo(f"key: {on_host(key_path)} — leave it shut until every segment is judged")


@main.command("caption-export")
@click.argument("caption_set", default="default", callback=_set_name)
@click.option("-o", "--output", default=None, type=UserPath(file_okay=False),
              help="Folder for the CSV [default: DATA_DIR/cuts/caption-export_<time>/].")
@click.pass_obj
def caption_export(data_dir, caption_set, output):
    """Write CAPTION_SET from the cache as `<CAPTION_SET>.csv`, for use outside the tool."""
    from .captions import CaptionCache

    if output:
        guard.output(output)
    cache = CaptionCache(data_dir, caption_set)
    if not cache.rows():
        raise click.ClickException(f"no captions in set {caption_set!r}; `mm status` lists the sets")
    path = _new_file(Path(output) if output else _cut_dir(data_dir, "caption-export"),
                     f"{caption_set}.csv")
    click.echo(f"wrote {on_host(path)} ({cache.export(path)} captions)")


@main.command()
@click.argument("query")
@click.option("-k", default=10, show_default=True, help="Max results.")
@click.option("--backend", default="siglip", show_default=True)
@click.option("--json", "as_json", is_flag=True,
              help="Print the hits as JSON; the project paths go to stderr.")
@click.option("-o", "--output", default=None, type=UserPath(file_okay=False),
              help="Folder for the LosslessCut projects "
                   "[default: DATA_DIR/cuts/<query>_<time>/]. They name their "
                   "video relative to this folder, so open them where they are.")
@click.option("--pad", default=1.0, show_default=True,
              help="Seconds added before and after each hit, in what is "
                   "printed and in the projects, so `mm cut` cuts what "
                   "LosslessCut shows.")
@click.option("--static/--moving", "static", default=None,
              help="Keep only locked-off shots, or only shots where most of "
                   "the frame changes. A filter, not a search term. Segments "
                   "with no motion value carry none and drop out.")
@click.option("--static-max", default=STATIC_MAX, show_default=True,
              help="A shot is static when fewer than this fraction of pixels "
                   "change between frames. Applied here, not at index time, "
                   "so retuning it costs nothing.")
@click.pass_obj
def search(data_dir, query, k, backend, as_json, output, pad, static, static_max):
    """Rank the moments matching QUERY, print them, and write one LosslessCut
    project per video with the hits as segments. Cuts nothing: `mm cut`
    does, once the projects look right."""
    from .embeddings import get_backend
    from .export import slug, write_llc_projects
    from .search import search as run_search

    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    hits = run_search(query, be, store, k=k, static=static, static_max=static_max)
    manifest = Manifest(data_dir / "manifest.db")
    for h in hits:
        row = manifest.video(h["path"])
        h["t0"] = max(0.0, h["t0"] - pad)
        h["t1"] = h["t1"] + pad if row is None else min(row["duration"], h["t1"] + pad)
    if as_json:
        click.echo(jsonlib.dumps(
            [{**{k_: h[k_] for k_ in ("t0", "t1", "score", "text")}, "path": on_host(h["path"])}
             for h in hits], indent=2))
    elif not hits:
        click.echo("no results")
    else:
        for h in hits:
            click.echo(f"{on_host(h['path'])}\n  {fmt_ts(h['t0'])} - {fmt_ts(h['t1'])}"
                       f"  (score {h['score']:.4f})")
            if h.get("text"):
                click.echo(f"  \"{h['text'][:120]}\"")
    if not hits:
        return
    if output:
        guard.output(output)
    written, taken = write_llc_projects(hits, output or _cut_dir(data_dir, slug(query)), query)
    for f in written:
        click.echo(f"project: {on_host(f)}", err=as_json)
    _end_run(taken=taken)


@main.command()
@click.argument("projects", nargs=-1, required=True,
                type=UserPath(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None, type=UserPath(file_okay=False),
              help="Folder for the clips [default: DATA_DIR/cuts/<project>_<time>/].")
@click.option("--smart/--no-smart", default=True, show_default=True,
              help="Frame-accurate cuts, recoding only the frames between each "
                   "cut and its nearest keyframe. --no-smart copies without "
                   "recoding and starts at the keyframe before (up to ~1 s early).")
@click.option("--re-encode", "re_encode", is_flag=True,
              help="Encode each whole clip again, exact at both ends. Every "
                   "setting below defaults to the source's own; timecode and "
                   "creation time are kept.")
@click.option("--size", default=None, callback=lambda c, p, v: v and _frame_size(c, p, v),
              help="With --re-encode: frame size WxH, e.g. 1920x1080.")
@click.option("--codec", default=None, type=click.Choice(["h264", "h265", "prores"]),
              help="With --re-encode: h264 for phones and the web, h265 smaller, "
                   "prores for editing (written as .mov).")
@click.option("--crf", default=None, type=click.IntRange(0, 51),
              help="With --re-encode: ffmpeg's constant quality, lower is better "
                   "and bigger (18 looks lossless). Without it the clip aims at the "
                   "source's bitrate. Not for prores.")
@click.option("--fps", default=None, type=click.FloatRange(min=0, min_open=True),
              help="With --re-encode: frame rate.")
@click.option("--no-audio", "no_audio", is_flag=True, help="With --re-encode: drop the sound.")
@click.pass_obj
def cut(data_dir, projects, output, smart, re_encode, size, codec, crf, fps, no_audio):
    """Cut every segment of each LosslessCut PROJECT (.llc) into a clip,
    exactly as the project says: one written by `mm search` or saved in
    LosslessCut. A segment ranked by `mm search` gets its rank in front of
    the clip's name. A taken name or a failed cut does not stop the rest."""
    from .export import Encode, cut_project

    encode = None
    if re_encode:
        if not smart:
            raise click.UsageError("--re-encode and --no-smart are two different cuts; pick one")
        if codec == "prores" and crf is not None:
            raise click.UsageError("prores has no --crf; leave it out")
        encode = Encode(size=size, codec=codec, crf=crf, fps=fps, audio=not no_audio)
    elif any(v not in (None, False) for v in (size, codec, crf, fps, no_audio)):
        raise click.UsageError("--size, --codec, --crf, --fps and --no-audio need --re-encode")
    if output:
        guard.output(output)
    manifest = Manifest(data_dir / "manifest.db")
    taken, failed = [], []
    for project in projects:
        out_dir = output or _cut_dir(data_dir, Path(project).stem)
        written, project_taken, project_failed = cut_project(project, out_dir, smart=smart,
                                                             encode=encode)
        for out, video, a0, a1 in written:
            manifest.add_export(out)
            click.echo(f"{on_host(out)}  <- {on_host(video)} {fmt_ts(a0)} - {fmt_ts(a1)}")
        taken += project_taken
        failed += project_failed
    _end_run(taken=taken, failed=failed)


@main.command()
@click.option("--labels", default=None, type=UserPath(dir_okay=False),
              help="Labels CSV (query,video,start,end) "
                   "[default: DATA_DIR/labels.csv, where `mm annotate` writes].")
@click.option("-k", default=10, show_default=True, help="Hits scored per query.")
@click.option("--backend", default="siglip", show_default=True)
@click.pass_obj
def eval(data_dir, labels, k, backend):
    """Score search quality against the labels."""
    from .embeddings import get_backend
    from .evaluate import evaluate, load_labels

    labels = Path(labels) if labels else data_dir / "labels.csv"
    if not labels.is_file():
        raise click.ClickException(f"no labels file at {on_host(labels)}")
    click.echo(f"labels: {on_host(labels)}")
    be = get_backend(backend)
    store = SegmentStore(data_dir, be.name)
    report = evaluate(load_labels(labels), be, store, k=k)
    for q in report["per_query"]:
        click.echo(f"  {q['found']}/{q['total']}  {q['query']}")
    click.echo(
        f"recall@{report['k']}: {report['recall']:.2f}"
        f"   MRR: {report['mrr']:.2f}   ({report['n_labels']} labels)"
    )


def _end_run(taken: list[str] = (), failed: list[str] = ()) -> None:
    """End a run that skipped or failed some files, so they are the last
    thing printed and the exit code says it."""
    lines = ([f"not written, the name was taken: {on_host(p)}" for p in taken]
             + [f"failed: {f}" for f in failed])
    if lines:
        raise click.ClickException(
            f"{len(lines)} file(s) not written:\n  " + "\n  ".join(lines))


def _new_file(folder: Path, name: str) -> Path:
    """`folder/name`, refused before anything is written if the name is taken."""
    path = guard.writable(Path(folder) / name)
    if path.exists():
        raise click.ClickException(f"not written, the name was taken: {on_host(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cut_dir(data_dir: Path, name: str) -> Path:
    """`<data-dir>/cuts/<name>_<time>/`, a new folder for each run; `name`
    must already be safe in a file name."""
    from datetime import datetime

    return data_dir / "cuts" / f"{name}_{datetime.now():%Y-%m-%d_%H-%M-%S}"


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
@click.argument("folder", type=UserPath(exists=True, file_okay=False))
@click.option("--template", "template_name", default=None,
              help="Start labels.csv from a template if it doesn't exist yet.")
@click.option("--labels", "labels_path", default=None, type=UserPath(dir_okay=False),
              help="Labels file [default: DATA_DIR/labels.csv].")
@click.pass_obj
def annotate(data_dir, folder, template_name, labels_path):
    """Interactively append ground-truth labels for videos in FOLDER.

    Labels go to DATA_DIR/labels.csv unless --labels names another file;
    the videos are only read. Watch the video in any player, then enter the
    times here.
    """
    from .manifest import VIDEO_EXTS

    if labels_path:
        guard.output(Path(labels_path).parent)
    labels = guard.writable(Path(labels_path) if labels_path else data_dir / "labels.csv")
    if not labels.exists():
        labels.parent.mkdir(parents=True, exist_ok=True)
        if template_name:
            src = pkg_files("moment_miner") / "templates" / f"labels_{template_name}.csv"
            if not src.is_file():
                raise click.BadParameter(
                    f"unknown template {template_name!r}; available: {_template_names()}")
            labels.write_text(src.read_text())
            click.echo(f"created {on_host(labels)} from template {template_name!r}")
        else:
            labels.write_text("query,video,start,end\n")
            click.echo(f"created empty {on_host(labels)}"
                       f" (templates available: {_template_names()})")
    videos = sorted(p for p in Path(folder).rglob("*")
                    if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        raise click.ClickException(f"no videos found under {on_host(folder)}")
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
    click.echo(f"done — {added} label(s) appended to {on_host(labels)}")
    if added:
        named = f" --labels {on_host(labels)}" if labels_path else ""
        click.echo(f"score the index anytime:  mm eval{named}")


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
    """Index statistics and the caption sets in the cache."""
    from .captions import CAPTIONS_DB, CaptionCache

    manifest = Manifest(data_dir / "manifest.db")
    for row in manifest.stats():
        click.echo(f"{row['status']:>8}: {row['n']} videos, {row['dur'] / 3600:.1f} h")
    if (data_dir / CAPTIONS_DB).exists():
        for name, n in CaptionCache(data_dir).sets():
            click.echo(f"captions: {n} in set {name!r}")
    if show_errors:
        for row in manifest.errors():
            click.echo(f"  {on_host(row['path'])}: {row['error']}")


@main.command(
    epilog="Rankers: " + ", ".join(_ranker_names()))
@click.argument("prompts", type=UserPath(exists=True, dir_okay=False))
@click.option("--captions", "caption_set", default=None,
              help="A caption set from the cache (`mm status` lists them). "
                   "Defaults to the axes already in the index.")
@click.option("--axis", default="caption", show_default=True,
              type=click.Choice(["caption", "action", "who", "scene", "light"]),
              help="Which axis each prompt is scored against.")
@click.option("--ranker", default="mock", show_default=True,
              help="How to score. See the list below.")
@click.option("-k", default=5, show_default=True, help="Hits shown per prompt.")
@click.option("--backend", default="siglip", show_default=True,
              help="Which index to read video names from.")
@click.pass_obj
def probe(data_dir, prompts, caption_set, axis, ranker, k, backend):
    """Run a file of free-form prompts against a caption set.

    `mm eval` scores the label set, which has ground truth and twelve rows.
    This answers the other question — whether the words you would actually
    type find the moment — and it has no ground truth, so it prints hits to
    read rather than a number to quote.
    """
    from .captions import CaptionCache
    from .rankers import get_ranker

    if caption_set:
        rows = CaptionCache(data_dir, caption_set).rows()
    else:
        rows = AxisStore(data_dir).all()
    if not rows:
        raise click.ClickException(
            f"no captions in set {caption_set!r}; `mm status` lists the sets" if caption_set
            else "no axes in the index — run `mm caption` first, or pass --captions")
    # A segment id is a hash, which answers nothing on its own. The index
    # knows which file it came from, so a hit names a video you can open.
    try:
        names = {s["id"]: Path(s["path"]).name
                 for s in SegmentStore(data_dir, backend).segments()}
    except Exception:
        names = {}
    try:
        r = get_ranker(ranker)
    except ValueError as e:
        raise click.UsageError(str(e)) from e
    queries = [q.strip() for q in open(prompts)
               if q.strip() and not q.startswith("#")]
    for q in queries:
        click.echo(f"\n{q}")
        for hit in r.top(q, rows, k=k, axis=axis):
            where = names.get(hit["id"], hit["id"].split(":")[0])
            click.echo(f"  {hit['score']:.3f}  {where} "
                       f"{fmt_ts(float(hit['t0']))} - {fmt_ts(float(hit['t1']))}  "
                       f"{hit.get(axis, '')}")


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
@click.option("-v", "--video", type=UserPath(exists=True, dir_okay=False),
              help="One video to search inside.")
@click.option("--labels", type=UserPath(exists=True, dir_okay=False),
              help="CSV of query,video,start,end — locate every row and shade the label.")
@click.option("--videos", type=UserPath(exists=True, file_okay=False),
              help="Where the videos named by --labels live.")
@click.option("--fps", default=5.0, show_default=True,
              help="Frames scored per second. The resolution knob, set per query.")
@click.option("--locator", "locator_name", default="maxsub", show_default=True,
              type=click.Choice(sorted(LOCATORS)))
@click.option("--duration", type=float,
              help="Region length in seconds. Required by --locator window, refused by maxsub.")
@click.option("-k", "top_k", default=1, show_default=True, help="Regions per video.")
@click.option("--backend", default="siglip", show_default=True)
@click.option("-o", "--output", type=UserPath(file_okay=False),
              help="Also write locate.html, a page of the score curves, into this folder.")
def locate(query, video, labels, videos, fps, locator_name, duration, top_k, backend, output):
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

    page = None
    if output:
        guard.output(output)
        page = _new_file(Path(output), "locate.html")
    be = get_backend(backend)
    loc = build_locator(locator_name, duration=duration, top_k=top_k)
    root = Path(videos) if videos else None
    panels = []
    for r in rows:
        path = r["video"] if root is None else next(
            (str(p) for p in root.rglob(r["video"])), None)
        if path is None:
            click.echo(f"  skipped {r['video']}: not found under {on_host(videos)}")
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

    if page:
        template = (pkg_files("moment_miner") / "templates" / "locate.html").read_text()
        page.write_text(template.replace(
            "__PANELS__", jsonlib.dumps(panels)).replace(
            "__META__", jsonlib.dumps({"fps": fps, "locator": locator_name,
                                       "duration": duration, "backend": be.name})))
        click.echo(f"wrote {on_host(page)} ({len(panels)} panel(s))")


if __name__ == "__main__":
    main()
