IMAGE   ?= moment-miner
VIDEOS  ?= examples
Q        ?= kong vault
K        ?= 10
OUT      ?= clips
DATA     ?= mm_data
SUB      ?= .
TEMPLATE ?= parkour
PORT     ?= 7700
GPUS     ?=

RUN = docker run --rm $(GPUS) \
	-v $(abspath $(VIDEOS)):/videos:ro \
	-v mm-hf-cache:/root/.cache \
	-v $(DATA):/data

.PHONY: image dev-image test index search mine annotate eval serve shell

image:
	docker build --target runtime -t $(IMAGE) .

dev-image:
	docker build -t $(IMAGE):dev .

test: dev-image
	docker run --rm $(IMAGE):dev

index: image
	$(RUN) $(IMAGE) index /videos/$(SUB)

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
	$(RUN) $(IMAGE) eval /videos/labels.csv

serve: image
	$(RUN) -p $(PORT):$(PORT) $(IMAGE) serve --host 0.0.0.0 --port $(PORT)

shell: dev-image
	$(RUN) -it --entrypoint bash $(IMAGE):dev
