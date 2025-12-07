.PHONY: help deploy-local deploy-aws destroy status port-forward-grafana port-forward-prometheus test lint clean

# Default target
help:
	@echo "AI FinOps Platform - Available Commands"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-local          Deploy to local Kind cluster"
	@echo "  make deploy-aws            Deploy to AWS EKS"
	@echo "  make destroy               Remove all resources"
	@echo "  make status                Check deployment status"
	@echo ""
	@echo "Development:"
	@echo "  make port-forward-grafana  Forward Grafana to localhost:3000"
	@echo "  make port-forward-prometheus Forward Prometheus to localhost:9090"
	@echo "  make test                  Run tests"
	@echo "  make lint                  Lint Python code"
	@echo "  make clean                 Clean build artifacts"
	@echo ""
	@echo "Build:"
	@echo "  make build                 Build container images"
	@echo "  make push                  Push images to registry"

# Deployment targets
deploy-local:
	@echo "Deploying to local Kind cluster..."
	./scripts/deploy-local.sh

deploy-aws:
	@echo "Deploying to AWS EKS..."
	kubectl apply -k deploy/kubernetes/overlays/aws

destroy:
	@echo "Removing AI FinOps Platform..."
	kubectl delete namespace ai-finops --ignore-not-found=true

status:
	@echo "=== AI FinOps Platform Status ==="
	@echo ""
	@echo "Namespace:"
	@kubectl get namespace ai-finops 2>/dev/null || echo "Namespace not found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -n ai-finops 2>/dev/null || echo "No pods found"
	@echo ""
	@echo "Services:"
	@kubectl get svc -n ai-finops 2>/dev/null || echo "No services found"

# Port forwarding
port-forward-grafana:
	@echo "Forwarding Grafana to localhost:3000..."
	@echo "Default credentials: admin / admin"
	kubectl port-forward -n ai-finops svc/grafana 3000:3000

port-forward-prometheus:
	@echo "Forwarding Prometheus to localhost:9090..."
	kubectl port-forward -n ai-finops svc/prometheus 9090:9090

# Development targets
test:
	@echo "Running tests..."
	cd src/calculator && python -m pytest tests/ -v

lint:
	@echo "Linting Python code..."
	cd src/calculator && ruff check . && ruff format --check .

lint-fix:
	@echo "Fixing lint issues..."
	cd src/calculator && ruff check --fix . && ruff format .

# Build targets
build:
	@echo "Building container images..."
	docker build -t ai-finops-calculator:latest -f src/calculator/Dockerfile src/calculator

push:
	@echo "Pushing images to registry..."
	docker tag ai-finops-calculator:latest ghcr.io/judeoyovbaire/ai-finops-calculator:latest
	docker push ghcr.io/judeoyovbaire/ai-finops-calculator:latest

# Clean targets
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
