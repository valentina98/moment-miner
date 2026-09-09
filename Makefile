IMAGE   ?= moment-miner
VIDEOS  ?= examples
Q        ?= kong vault
K        ?= 10
OUT      ?= clips
DATA     ?= mm_data
SUB      ?= .
# Both are paths under VIDEOS: SUB is the folder to index, LABELS the eval set.
# Ground truth lives beside the footage, so `eval` wants VIDEOS=footage while
# `index` wants VIDEOS=footage/src.
LABELS   ?= labels.csv
ARGS     ?=
TEMPLATE ?= parkour
PORT     ?= 7700
CAPTION_MODEL ?= claude-haiku-4-5
GPUS     ?=

# Captioning takes either route: ANTHROPIC_API_KEY, or the Claude Code
# subscription token that resolve_credentials() reads from .credentials.json.
# The token route only works if the file is inside the container, so mount it
# when it exists -- absent, this expands to nothing and the API key path is
# unaffected.
CREDS = $(wildcard $(HOME)/.claude/.credentials.json)
CREDS_MOUNT = $(if $(CREDS),-v $(CREDS):/root/.claude/.credentials.json:ro,)

RUN = docker run --rm $(GPUS) \
	-e ANTHROPIC_API_KEY \
	$(CREDS_MOUNT) \
	-v $(abspath $(VIDEOS)):/videos:ro \
	-v mm-hf-cache:/root/.cache \
	-v $(DATA):/data

# Captions are cached beside the footage, so this one target mounts the archive
# read-write. Everything else keeps :ro.
RUN_RW = $(subst :ro,,$(RUN))

.PHONY: image dev-image test index caption search mine annotate eval serve shell

image:
	docker build --target runtime -t $(IMAGE) .

dev-image:
	docker build -t $(IMAGE):dev .

test: dev-image
	docker run --rm $(IMAGE):dev

index: image
	$(RUN) $(IMAGE) index /videos/$(SUB) $(ARGS)

caption: image
	$(RUN_RW) $(IMAGE) index /videos/$(SUB) --caption $(CAPTION_MODEL)

search: image
	$(RUN) $(IMAGE) search "$(Q)" -k $(K) --llc-dir /data/llc

mine: image
	mkdir -p $(OUT)
	$(RUN) -v $(abspath $(OUT)):/out $(IMAGE) mine "$(Q)" /videos/$(SUB) -o /out -k $(K)

# interactive; footage mounted writable because labels.csv lives with it
annotate: image
	docker run --rm -it \
		-v $(abspath $(VIDEOS)):/videos \
		-v mm-hf-cache:/root/.cache -v $(DATA):/data \
		$(IMAGE) annotate /videos --template $(TEMPLATE)

eval: image
	$(RUN) $(IMAGE) eval /videos/$(LABELS)

serve: image
	$(RUN) -p $(PORT):$(PORT) $(IMAGE) serve --host 0.0.0.0 --port $(PORT)

shell: dev-image
	$(RUN) -it --entrypoint bash $(IMAGE):dev
