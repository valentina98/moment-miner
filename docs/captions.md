# Captioning segments

`mm index --caption <model>` writes one sentence per segment into the searchable text. This is what makes a silent clip findable by word. `mm caption FOLDER` does the same for segments already indexed, without re-embedding them.

## Why

`search()` fuses two rankings with RRF: vector similarity over the segment embeddings, and full-text search over the `text` column (`moment_miner/search.py:38-42`). That column is filled from the Whisper transcript and nothing else (`moment_miner/indexer.py`), and it carries the FTS index (`moment_miner/store.py:52`).

Footage without speech leaves that column empty, so the text half of the fusion contributes nothing and search is effectively vision-only. Captions populate it — "pull-up" or "foam pit" become findable by word rather than only by pixel similarity. Captions are appended to the same column: no schema change, no change to the fusion.

## Where captions live

An archive is self-contained: everything derived from a folder of footage is written back into that folder, so a drive carries its own captions and can be unplugged and read elsewhere. `--caption-dir` and `-o` override that when the archive has to stay read-only.

`<folder>/captions.csv`, beside the footage — the same convention as the `labels.csv` that `mm eval` reads.

They are separate files on purpose. **Captions are machine-written indexed content; labels are human-checked ground truth.** Scoring captions against labels written by the same model measures agreement, not accuracy, so a caption is never also the label it is scored against.

Columns: `id,t0,t1,caption,model,written`.

Captions are read back on the next pass. Re-indexing a folder — including `--reindex` — reuses the captions already on disk and only spends on segments that don't have one. This is why they live next to the footage rather than in `mm_data`: a caption costs session quota (money only on the API-key route), and a deleted volume means a re-index, not a second caption pass.

## The method

**Segment geometry.** 8-second window, 4-second stride (`moment_miner/indexer.py`, overridable with `--window` / `--stride`). Segments start at 0 and step forward a stride at a time; a leftover under half a stride stretches the last segment to the video's end, and a larger one gets its own full-length segment ending there. So every segment is full length and the video's final frame is the final frame of a segment — which is exactly where the frame sampler looks.

**Frames.** How many stills a segment gets follows from its span and the stride, not from a fixed number. Under one stride: one frame, in the middle. Under two strides: two, at a third and two thirds. Otherwise: anchored half a stride from the end and stepping back a whole stride at a time, as many as fit. That is one frame per stride-length block of video, at the block's midpoint — and since windows overlap by half, neighbouring windows agree on the frame they share. At the defaults — 8 s window, 4 s stride — an ordinary segment yields two frames, at 2 s and 6 s. The anchor is at the end because that is where the action resolves: the trick lands and recording stops.

These tiers were reasoned about at stride 4 and do not adapt to another one — at stride 2 a full window would yield four frames instead of two, and frames are the dominant cost of a pass. `--caption-frames` forces a count and overrides the rule; `--caption-frames-short` / `--caption-frames-long` do the same per video length.

**Prompt.** The `PROMPT` constant in `moment_miner/captions.py` is the single source of truth. It asks for a **label of 10 to 20 words, not a description**, in the plain words someone would type into a search box — "cutting the cake, two people, church hall with wooden beams, indoors". Sentences, framing ("a video showing"), and commentary on the shot are all banned, and a subject too small to identify must be called unclear rather than guessed. What the label names, and why those things and not others, is the next section.

A longer budget turns the label into a description: *"Graffiti-covered concrete river embankment under a road bridge, people leaning on the railing above, green water and wooded hillside; the static shot barely changes."* Nobody types that, and against a BM25 index the extra words dilute the terms that matter.

## What the label carries, and how to change it

**The label is the retrieval surface: what it names is what can be found.** No ranker recovers an axis the caption never mentions — not BM25, not an embedding, not a reranker. So this list is a choice about which searches are possible, and it is meant to be revisited when the task changes.

| Axis | Why it is in |
| --- | --- |
| Action | The thing searched for, almost always. It leads, so its words carry the most weight. |
| Who is in the shot | Counts discriminate — two people talking is not one person working. A word or two. |
| Surroundings | Named concretely. "Outdoors" alone is what makes many clips from one session read alike. |
| Light | Cheap, visibly varies, not recoverable from metadata. |

Three things are left out on purpose. **Outcome** — clean landing, bail, slip — because a handful of stills cannot show a stumble, so asking for it invites an invented one. **Shot type**, as the least certain thing to read off stills. **Who someone is, by name**: a vision model should not guess identity, so names belong in a file beside the footage, joined into the segment's text the way a transcript is.

**The prompt is a template, and two ship.** They live in `moment_miner/templates/` as `caption_<name>.txt` and are chosen with `--caption-prompt`, which also takes a path to your own. `general` is the one to copy: it names the four axes with examples from no particular world. `parkour` is a worked example of the same template tuned to one corpus, and it shows what tuning is for — on parkour footage filmed at a skate spot, the general prompt returns "skateboarding", "wheelie" and "bicycle stunt", while the tuned one returns vaults and wall runs from the same stills. A prompt is a template because what a caption names decides what can be searched, and that differs per corpus.

**What the index already measures exactly does not belong in a label.** Static-vs-moving, frame rate, duration and date are numbers, held in the motion table and the probe metadata; a caption guessing "static shot" duplicates a measurement and spends words doing it. Those are filters — the label is for what only a reader of the frames can say.

**To re-evaluate the axes:** write down the queries the new task actually types, before looking at any caption; name the axis each one searches on, since an axis no caption names is a search that cannot work however good the ranker is; then change `PROMPT` and re-caption to a separate `--caption-dir`, so the old set survives as a reference to compare against. Each axis costs two to four words, and a longer label dilutes every term in it. Caption sets written by different prompts are different instruments — re-run a comparison rather than reading across.

## Credentials

Resolved in this order, once, when the first caption is requested. The free route wins, and the paid one is refused unless it was asked for and capped.

1. Claude Code's subscription token: `claudeAiOauth.accessToken` in `$CLAUDE_CONFIG_DIR/.credentials.json` (default `~/.claude`), and nothing else in that file, which also holds refresh tokens and MCP servers' tokens. Under Docker the file never enters the container: `make caption` and `make recaption` copy that one value into a temp file, mount it read-only and delete it afterwards, and no other target gets a token — spends session quota, no money
2. `ANTHROPIC_API_KEY` — spends money, so it is used **only** with `--allow-paid` and `--paid-budget-usd` above zero. A key on its own, even with no subscription credentials present, raises rather than bills
3. Neither — one error naming both, raised before any frame is sent

The budget is a gate, not a meter: it is checked before the first request and nothing counts dollars as the pass runs. Metering is a next step, not a feature this claims.

Captioning is the only stage that spends anything — money on an API key, session quota on a subscription. `mm index` without `--caption` needs no credentials at all.

## What it costs

**Nothing, on the subscription route.** The Claude Code token spends session quota and no money, and it is the route both `make caption` and `make recaption` take. The figures below are what the same work would cost through `ANTHROPIC_API_KEY`, which is only reachable with `--allow-paid` and a budget.

Segments per hour of footage is `3600 / stride` — 900 at the default stride 4, and higher on an archive of short clips, whose truncated windows do not overlap (a 50-clip corpus measured 1086). A still at `mm caption`'s default 640x360 is ~300 input tokens (299 by counting 28-pixel tiles, 307 by `w x h / 750`; confirm with `count_tokens` before a bulk run), the prompt adds ~200, and ~30 output tokens come back. So an ordinary two-still segment is ~800 in and ~30 out, and an hour of footage is **0.72 M input, 0.027 M output**. Under `mm index --caption` stills are the 456x256 the embedding already decoded, ~170 tokens each, which is about 0.6x of that.

**Frame count is the main cost lever** — each extra still adds ~300 tokens to an ~800-token request, so forcing 5 frames costs roughly 1.9x what the rule's two do.

| Model | $/MTok in / out | Per hour of footage | Batch API (-50%) |
| --- | --- | --- | --- |
| Opus 5 | $5 / $25 | $4.28 | $2.14 |
| Sonnet 5 | $2 / $10 | $1.71 | $0.86 |
| Haiku 4.5 | $1 / $5 | $0.86 | $0.43 |

A small corpus makes that concrete: 131 segments is ~105 k input and ~4 k output tokens, about **$0.12** on Haiku 4.5 — and $0 on the subscription.

Prices read from Anthropic's model pricing 2026-06-24; re-check before a bulk run. Prompt caching does not help — every image is unique. The Batch API's 50% does, and it is a price discount, so it buys nothing on the subscription.

## Running it

```bash
mm index /videos --caption claude-haiku-4-5
mm index /videos --caption claude-opus-5 --caption-frames-long 6
make caption VIDEOS=footage CAPTION_MODEL=claude-haiku-4-5
mm caption /videos --model claude-haiku-4-5 --frame-size 640x360
make recaption VIDEOS=footage/src CAPTION_MODEL=claude-haiku-4-5 ARGS="--force"
```

**Two routes.** `mm index --caption` captions while it indexes, from the frames it already decoded for the embedding, so the model never sees more than 456x256. `mm caption` is a separate pass over an index that already exists: it reads each segment's span from the store, seeks to the same instants `index --caption` would pick, decodes only those stills at `--frame-size` (default 640x360; anything wider than 640 px is downscaled before sending, `FRAME_WIDTH` in `captions.py`), then rewrites the row's `text` as transcript plus caption and rebuilds the FTS index. Vectors, schema and fusion are untouched, and the embedding model is never loaded. Use it when the caption model, prompt or raster changes and the embeddings have not, which costs no re-index. Segments that already have a sidecar caption are not bought again, but the cached caption is still written into the row; `--force` buys a fresh one. Stride and frames-per-window are read from what the manifest recorded for each video.

`make caption` exists because captions are written beside the footage: it mounts the archive read-write, where every other target mounts it `:ro`. Running `mm index --caption` against a read-only archive fails immediately, before the first paid request, rather than after. `--caption-dir` writes the sidecars elsewhere if the archive must stay read-only.

## Comparing caption sets

```bash
mm caption-compare opus.csv sonnet.csv haiku.csv --videos /videos
```

Renders one HTML page: each segment's stills, the captions under them, and a mark where the models disagree. Written to `caption-comparison.html` inside `--videos` — on the archive, with the footage it describes.

**Blind by default.** Which file wrote which caption is hidden and the column order is shuffled, because you necessarily know which model you launched; the blinding has to happen at judging time. The order is *balanced*, not merely random — every ordering used equally often, so each set sits in each column the same number of times. A free shuffle put one set in the middle column on 10 of 18 segments, which is exactly the position bias the shuffle exists to remove. The decode map is written beside the page as `<name>-key.csv`; leave it shut until every segment is judged. `--no-blind` labels the columns by filename instead.

**Disagreement is marked where the captions share no content word at all.** That is a fact about the text rather than a tuned threshold — and on the first real bake-off it marked 9 of 18 segments, where a graded score marked 17 and so said nothing. Words appearing in only one caption are highlighted inline: that is where one model saw a dam wall and another a bridge.

The test is lexical, so it has two known failure directions: a paraphrase ("bails a trick" / "fails a landing") reads as disagreement, and a confident error all the models share reads as agreement. It is a reading aid for a human judging captions, not a metric — nothing in the index depends on it.

`--frames-dir` uses stills already extracted elsewhere, named `<segment id>_*.jpg`, instead of decoding the videos. `--segments` supplies timings when the caption files carry only `seg,caption`.

## What captions get wrong

- **Stills lose motion and sound.** A swing versus a hang, a failed landing, someone talking — three frames per window is where that error enters.
- **Small subjects.** In wide 4K shots a subject can be 2% of frame height. The prompt tells the model to say so; a caption that confidently names an action at that scale is the failure to look for.
- **Repetition.** Many clips from one session share scenery. A model that writes near-identical captions for different moments is useless for retrieval even when each caption reads well.
