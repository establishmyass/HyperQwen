# HyperQwen shortcuts. Every target delegates to the local container engine or a
# checked-in script — no hidden flags. `make help` lists them.
# Native (venv) launchers remain: bash single-user/start_qwen.sh etc.

ENGINE ?= $(if $(shell command -v docker),docker,podman)
COMPOSE ?= $(ENGINE) compose
# Podman Compose 1.6 requires named profiles. Its ps output is scoped to this project.
ifeq ($(ENGINE),podman)
ACTIVE_PROFILES = $(addprefix --profile=,$(sort $(shell $(COMPOSE) ps --format '{{.Label "com.docker.compose.service"}}' 2>/dev/null)))
else
ACTIVE_PROFILES = --profile '*'
endif

.PHONY: help up up-single up-batch up-inkling up-inkling-4090 up-inkling-4090-dspark up-inkling-4090-balanced down logs ps verify verify-install shell keygen doctor

help: ## show this list
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

up: up-inkling-4090-balanced ## serve Inkling with balanced prefill and DSpark decode on one RTX 4090

up-single: ## serve one/few users (SPEC=dflash2 profile from .env)
	$(COMPOSE) --profile single up -d
	@echo "first boot compiles (~2-15 min); follow with: make logs"

up-batch: ## serve API backend / many concurrent
	$(COMPOSE) --profile batch up -d
	@echo "first boot compiles (~2-15 min); follow with: make logs"

up-inkling: ## serve Inkling 2 Mini on one supported GPU
	$(COMPOSE) --profile inkling up -d

up-inkling-4090: ## serve Inkling 2 Mini on one 24 GB RTX 4090
	$(COMPOSE) --profile inkling-4090 up -d

up-inkling-4090-dspark: ## serve Inkling with seven-token speculative decoding on one RTX 4090
	$(COMPOSE) --profile inkling-4090-dspark up -d

up-inkling-4090-balanced: ## serve Inkling with 1280-token prefill chunks and seven draft tokens
	$(COMPOSE) --profile inkling-4090-balanced up --build -d

down: ## stop server
	$(COMPOSE) $(ACTIVE_PROFILES) down

logs: ## follow server logs (Ctrl-C anytime)
	$(COMPOSE) $(ACTIVE_PROFILES) logs -f --tail 30

ps: ## container state + /health from inside WSL/network namespace
	$(ENGINE) ps --filter name=hyperqwen --format "{{.Names}} {{.Status}}"
	@curl -s -o /dev/null -w "health: HTTP %{http_code}\n" http://127.0.0.1:$${PORT:-18020}/health || true

verify: ## verify.sh inside the running single/batch container (live checks included), else in a fresh one (no live server)
	@svc=$$($(COMPOSE) ps --status running --services 2>/dev/null | grep -xE 'single|batch' | head -n1); \
	if [ -n "$$svc" ]; then echo "verify: running inside the running '$$svc' service"; $(COMPOSE) exec $$svc bash verify.sh; \
	else echo "verify: no single/batch service running; live-server checks will be skipped"; $(COMPOSE) run --rm single verify; fi

verify-install: ## install-only checks (venv, vLLM, patches, KVarN): no GPU/model/server
	bash verify.sh --install

shell: ## throwaway shell in the server image (patches + venv, no GPU boot)
	$(COMPOSE) run --rm single bash

keygen: ## append a fresh VLLM_API_KEY to .env without clobbering it (safe to re-run: skips if set)
	@if grep -q '^VLLM_API_KEY=.\+' .env 2>/dev/null; then echo ".env already has VLLM_API_KEY (presence-only check, value not shown)"; else KEY=$$(openssl rand -hex 24); printf 'VLLM_API_KEY=%s\n' "$$KEY" >> .env; echo "appended VLLM_API_KEY to .env"; fi

doctor: ## read-only status: config + GPU + port + /health + last log error (no secrets printed)
	bash scripts/hq-doctor.sh
