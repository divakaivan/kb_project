DOCKER_COMPOSE_FILE=docker-compose.yaml

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help message
	@echo ""
	@echo "Usage: make [option]"
	@echo ""
	@echo "Options:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

.PHONY: build
build:  ## Build docker services
	docker compose -f $(DOCKER_COMPOSE_FILE) build

.PHONY: start
start:  ## Start docker services (detached mode)
	docker compose -f $(DOCKER_COMPOSE_FILE) up -d

.PHONY: model-dict
model-dict:  ## Create model dictionary UI
	pip install -r model-dict/requirements.txt --quiet && streamlit run model-dict/app.py

.PHONY: stop
stop:  ## Stop docker services
	docker compose -f $(DOCKER_COMPOSE_FILE) stop
