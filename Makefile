IMAGE   ?= moment-miner
VIDEOS  ?= examples
Q        ?= kong vault
K        ?= 10
OUT      ?= clips
# Both take a docker volume name or an absolute host path, so an index and the
# model weights can live on the drive that holds the footage.
DATA     ?= mm_data
CACHE    ?= mm-hf-cache
SUB      ?= .
# Both are paths under VIDEOS: SUB is the folder to index, LABELS the eval set
# (or LABELS=/data/labels.csv for the ones `make annotate` wrote).
# The recorded eval set sits beside the footage, so `eval` wants VIDEOS=footage
# while `index` wants VIDEOS=footage/src.
LABELS   ?= labels.csv
# One prompt per line. Relative to the repo, not to VIDEOS, which often points
# at an external drive.
PROMPTS  ?= prompts.txt
AXIS     ?= action
RANKER   ?= mock
ARGS     ?=
# LLC=1: `search` also writes LosslessCut projects, into DATA/cuts/<query>_<time>/.
LLC      ?=
# Lets those projects name their video by its host path; only a DATA that is a
# host folder can be opened from the host at all.
HOST_PATHS = $(if $(filter /%,$(DATA)),-e MM_HOST_PATHS="/videos=$(abspath $(VIDEOS));/data=$(DATA)",)
TEMPLATE ?= parkour
PORT     ?= 7700
CAPTION_MODEL ?= claude-haiku-4-5
# Caption prompt: a shipped name (general, parkour) or a file path.
CAPTION_PROMPT ?= general
GPUS     ?=

# The subscription token reaches only `caption` and `recaption`, and never as
# the credentials file itself: that file also holds refresh tokens and every MCP
# server's OAuth token. The host copies the one value resolve_credentials()
# reads, claudeAiOauth.accessToken, into a private temp file (or `{}` when there
# is none, leaving the API key route to decide), mounts it read-only, and
# deletes it however the run ends.
CLAUDE_CREDS ?= $(HOME)/.claude/.credentials.json
TOKEN_FILE = tok=$$(mktemp) && trap 'rm -f "$$tok"' EXIT && trap 'exit 130' INT TERM && \
	python3 -c 'import json, os, sys; p = sys.argv[1]; e = (json.load(open(p)) if os.path.exists(p) else {}).get("claudeAiOauth") or {}; t = e.get("accessToken"); json.dump({"claudeAiOauth": {"accessToken": t}} if isinstance(t, str) and t else {}, sys.stdout)' \
	"$(CLAUDE_CREDS)" > "$$tok" &&
TOKEN_MOUNT = -v "$$tok":/root/.claude/.credentials.json:ro

# Keys come from .env when it exists, so a fresh checkout needs no exports;
# docker lets an exported variable of the same name win over the file.
RUN = docker run --rm $(GPUS) \
	$(if $(wildcard .env),--env-file .env,) \
	-e ANTHROPIC_API_KEY \
	-e TYPESAFE_API_KEY \
	$(HOST_PATHS) \
	-v $(abspath $(VIDEOS)):/videos:ro \
	-v $(CACHE):/root/.cache \
	-v $(DATA):/data

.PHONY: image dev-image test calibrate index caption recaption search mine annotate eval probe serve shell

image:
	docker build --target runtime -t $(IMAGE) .

dev-image:
	docker build -t $(IMAGE):dev .

test: dev-image
	docker run --rm $(IMAGE):dev

# Measures this machine once on 24 s of SUB (a video), so `index` can print an ETA; index runs it itself when missing.
calibrate: image
	$(RUN) $(IMAGE) calibrate /videos/$(SUB) $(ARGS)

index: image
	$(RUN) $(IMAGE) index /videos/$(SUB) $(ARGS)

caption: image
	$(TOKEN_FILE) $(RUN) $(TOKEN_MOUNT) $(IMAGE) index /videos/$(SUB) --caption $(CAPTION_MODEL) --caption-prompt $(CAPTION_PROMPT) $(ARGS)

# Captions an index that already exists, without re-embedding it.
recaption: image
	$(TOKEN_FILE) $(RUN) $(TOKEN_MOUNT) $(IMAGE) caption /videos/$(SUB) --model $(CAPTION_MODEL) --caption-prompt $(CAPTION_PROMPT) $(ARGS)

search: image
	$(RUN) $(IMAGE) search "$(Q)" -k $(K) $(if $(LLC),--llc,)

mine: image
	mkdir -p $(OUT)
	$(RUN) -v $(abspath $(OUT)):/out $(IMAGE) mine "$(Q)" /videos/$(SUB) -o /out -k $(K)

# Interactive; labels go to DATA/labels.csv.
annotate: image
	$(RUN) -it $(IMAGE) annotate /videos --template $(TEMPLATE)

eval: image
	$(RUN) $(IMAGE) eval $(if $(filter /data/%,$(LABELS)),$(LABELS),/videos/$(LABELS))

probe: image
	@test -f "$(PROMPTS)" || { echo "no prompts file at $(PROMPTS) -- write one, or pass PROMPTS=<path>"; exit 1; }
	$(RUN) -v $(abspath $(PROMPTS)):/prompts.txt:ro $(IMAGE) probe /prompts.txt --axis $(AXIS) --ranker $(RANKER) -k $(K) $(ARGS)

serve: image
	$(RUN) -p $(PORT):$(PORT) $(IMAGE) serve --host 0.0.0.0 --port $(PORT)

shell: dev-image
	$(RUN) -it --entrypoint bash $(IMAGE):dev
