IMAGE   ?= moment-miner
VIDEOS  ?= examples
Q        ?= kong vault
K        ?= 10
# Where `search` writes its projects and `cut` its clips, any host folder, a
# footage folder included [default: DATA/cuts/<name>_<time>/].
OUT      ?=
# The .llc project `cut` cuts: a host path under DATA, VIDEOS or OUT, the
# folders the container sees. Its video must be under VIDEOS.
PROJECT  ?=
# Both take a docker volume name or an absolute host path, so an index and the
# model weights can live on the drive that holds the footage. Both are host
# folders by default: labels, clips and .llc projects are opened from DATA, and
# a container running as you can write to neither volume docker created as root.
DATA     ?= $(HOME)/.mm_data
CACHE    ?= $(HOME)/.cache/moment-miner
SUB      ?= .
# SUB is the folder under VIDEOS a command works on (a video, for `calibrate`).
# LABELS is the labels file `annotate` writes and `eval` scores, a host path under
# DATA or VIDEOS [default: DATA/labels.csv].
LABELS   ?=
# One prompt per line. Relative to the repo, not to VIDEOS, which often points
# at an external drive.
PROMPTS  ?= prompts.txt
AXIS     ?= action
RANKER   ?= mock
ARGS     ?=
# Host paths let the tool print and take the paths you use, let projects name
# their video from the host, and let the tool see that an OUT inside VIDEOS is
# the same place, so it never indexes its own clips. Only a DATA that is a host
# folder has a host path.
HOST_PATHS = -e MM_HOST_PATHS="/videos=$(abspath $(VIDEOS))$(if $(OUT),;/out=$(abspath $(OUT)),)$(if $(filter /%,$(DATA)),;/data=$(DATA),)"
OUT_OPT = $(if $(OUT),-o $(abspath $(OUT)),)
LABELS_OPT = $(if $(LABELS),--labels $(abspath $(LABELS)),)
PORT     ?= 7700
# Left empty, these leave the option to PROFILE or to the command's own default.
TEMPLATE ?=
CAPTION_MODEL ?=
# Caption prompt: a shipped name (general, parkour) or a file path.
CAPTION_PROMPT ?=
# A profile (shipped: parkour), or your own in DATA/profiles/<name>.yaml: every
# target passes it, and it sets the defaults of that command's options.
PROFILE  ?=
PROFILE_OPT = $(if $(PROFILE),--profile $(PROFILE),)
GPUS     ?=

# The subscription token reaches only `caption`, and never as
# the credentials file itself: that file also holds refresh tokens and every MCP
# server's OAuth token. The host copies the one value resolve_credentials()
# reads, claudeAiOauth.accessToken, into a private temp file (or `{}` when there
# is none, leaving the API key route to decide), mounts it read-only, and
# deletes it however the run ends.
CLAUDE_CREDS ?= $(HOME)/.claude/.credentials.json
TOKEN_FILE = tok=$$(mktemp) && trap 'rm -f "$$tok"' EXIT && trap 'exit 130' INT TERM && \
	python3 -c 'import json, os, sys; p = sys.argv[1]; e = (json.load(open(p)) if os.path.exists(p) else {}).get("claudeAiOauth") or {}; t = e.get("accessToken"); json.dump({"claudeAiOauth": {"accessToken": t}} if isinstance(t, str) and t else {}, sys.stdout)' \
	"$(CLAUDE_CREDS)" > "$$tok" &&
TOKEN_MOUNT = -v "$$tok":/tmp/.claude/.credentials.json:ro

# For targets that call no hosted model, so they never hold a key. Runs as you,
# so what lands in DATA and OUT is yours rather than root's; that user has no
# home in the image, hence HOME=/tmp and the cache at /cache. A host folder is
# created first, or docker would create it owned by root.
RUN_NO_KEYS = $(if $(filter /%,$(DATA) $(CACHE) $(abspath $(OUT))),mkdir -p $(filter /%,$(DATA) $(CACHE) $(abspath $(OUT))) &&,) \
	docker run --rm $(GPUS) \
	--user $(shell id -u):$(shell id -g) -e HOME=/tmp -e XDG_CACHE_HOME=/cache \
	$(HOST_PATHS) \
	-v $(abspath $(VIDEOS)):/videos:ro \
	-v $(CACHE):/cache \
	-v $(DATA):/data \
	$(if $(OUT),-v $(abspath $(OUT)):/out,)
# Keys come from .env when it exists, so a fresh checkout needs no exports;
# docker lets an exported variable of the same name win over the file.
RUN = $(RUN_NO_KEYS) \
	$(if $(wildcard .env),--env-file .env,) \
	-e ANTHROPIC_API_KEY \
	-e TYPESAFE_API_KEY

.PHONY: image dev-image test calibrate index caption search cut annotate eval status probe serve shell

image:
	docker build --target runtime -t $(IMAGE) .

dev-image:
	docker build -t $(IMAGE):dev .

test: dev-image
	docker run --rm $(IMAGE):dev

# Each target runs the mm command of the same name; ARGS adds options to any of them.
# Measures this machine once on 24 s of SUB (a video), so `index` can print an ETA; index runs it itself when missing.
calibrate: image
	$(RUN_NO_KEYS) $(IMAGE) calibrate /videos/$(SUB) $(PROFILE_OPT) $(ARGS)

index: image
	$(RUN_NO_KEYS) $(IMAGE) index /videos/$(SUB) $(PROFILE_OPT) $(ARGS)

# Captions an index that already exists; the only target that can spend.
caption: image
	$(TOKEN_FILE) $(RUN) $(TOKEN_MOUNT) $(IMAGE) caption /videos/$(SUB) $(if $(CAPTION_MODEL),--model $(CAPTION_MODEL),) $(if $(CAPTION_PROMPT),--caption-prompt $(CAPTION_PROMPT),) $(PROFILE_OPT) $(ARGS)

search: image
	$(RUN_NO_KEYS) $(IMAGE) search "$(Q)" -k $(K) $(OUT_OPT) $(PROFILE_OPT) $(ARGS)

cut: image
	@test -n "$(PROJECT)" || { echo "name the project: make cut PROJECT=<path to a .llc file>"; exit 1; }
	$(RUN_NO_KEYS) $(IMAGE) cut "$(abspath $(PROJECT))" $(OUT_OPT) $(PROFILE_OPT) $(ARGS)

# Interactive.
annotate: image
	$(RUN_NO_KEYS) -it $(IMAGE) annotate /videos/$(SUB) $(if $(TEMPLATE),--template $(TEMPLATE),) $(LABELS_OPT) $(PROFILE_OPT) $(ARGS)

eval: image
	$(RUN_NO_KEYS) $(IMAGE) eval $(LABELS_OPT) $(PROFILE_OPT) $(ARGS)

status: image
	$(RUN_NO_KEYS) $(IMAGE) status $(PROFILE_OPT) $(ARGS)

# Keys only for the hosted ranker.
probe: image
	@test -f "$(PROMPTS)" || { echo "no prompts file at $(PROMPTS) -- write one, or pass PROMPTS=<path>"; exit 1; }
	$(if $(filter jev,$(RANKER)),$(RUN),$(RUN_NO_KEYS)) -v "$(abspath $(PROMPTS))":/prompts.txt:ro $(IMAGE) probe /prompts.txt --axis $(AXIS) --ranker $(RANKER) -k $(K) $(PROFILE_OPT) $(ARGS)

serve: image
	$(RUN_NO_KEYS) -p $(PORT):$(PORT) $(IMAGE) serve --host 0.0.0.0 --port $(PORT) $(PROFILE_OPT) $(ARGS)

shell: dev-image
	$(RUN) -it --entrypoint bash $(IMAGE):dev
