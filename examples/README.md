# Exemplar clips

**Currently empty, on purpose.** Nothing in the tool reads this directory yet; it is reserved, and this note exists so the reservation is legible rather than looking like an oversight.

## What goes here

One folder per move or event, holding a few short clips that show what it looks like:

```
examples/kong_vault/clip1.mp4
examples/swing_gainer/clip1.mp4
```

## What they are for

Two jobs, and they are not the same:

- **Vocabulary.** A captioner writes the words it was given. A move it has no name for comes back as "backflip" or "jump", and no text ranker can then answer a search for that move by name. Naming the moves in the caption prompt is the cheap half of this, and it only works for moves you can list in advance.
- **In-context examples for a vision model.** Showing a model what a move looks like, rather than naming it, is the half a word list cannot do. That is what the clips here are for.

## The rule that constrains them

**An exemplar must never cover a moment that `footage/labels.csv` also marks.** The labels are the eval set, so an exemplar drawn from one is the answer handed to the model in advance, and a with-exemplars run then scores higher for free. Take exemplars from unlabelled stretches of footage, or from clips the label set does not use at all.

Clips here are git-ignored. This note is the one file in the directory that is tracked.
