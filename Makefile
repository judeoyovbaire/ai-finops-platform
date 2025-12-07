.PHONY: help deploy-local deploy-aws destroy status port-forward-grafana port-forward-prometheus port-forward-opencost test lint clean

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
	@echo "  make port-forward-grafana     Forward Grafana to localhost:3000"
	@echo "  make port-forward-prometheus  Forward Prometheus to localhost:9090"
	@echo "  make port-forward-opencost    Forward OpenCost UI to localhost:9091"
	@echo "  make port-forward-enricher    Forward GPU Enricher API to localhost:8080"
	@echo "  make test                     Run tests"
	@echo "  make lint                     Lint Python code"
	@echo "  make lint-fix                 Fix lint issues"
	@echo "  make clean                    Clean build artifacts"
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

port-forward-opencost:
	@echo "Forwarding OpenCost UI to localhost:9091..."
	kubectl port-forward -n ai-finops svc/opencost 9091:9090

port-forward-enricher:
	@echo "Forwarding GPU Enricher API to localhost:8080..."
	@echo "API endpoints:"
	@echo "  GET /health           - Health check"
	@echo "  GET /metrics          - Prometheus metrics"
	@echo "  GET /api/v1/costs/summary - Cost summary by team"
	@echo "  GET /api/v1/recommendations - Optimization recommendations"
	@echo "  GET /api/v1/gpu/utilization - GPU utilization metrics"
	kubectl port-forward -n ai-finops svc/gpu-enricher 8080:8080

# Development targets
test:
	@echo "Running tests..."
	cd src/gpu-enricher && python -m pytest test_main.py -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	cd src/gpu-enricher && python -m pytest test_main.py -v --cov=main --cov-report=term-missing

lint:
	@echo "Linting Python code..."
	cd src/gpu-enricher && ruff check . && ruff format --check .

lint-fix:
	@echo "Fixing lint issues..."
	cd src/gpu-enricher && ruff check --fix . && ruff format .

# Build targets
build:
	@echo "Building container images..."
	docker build -t ai-finops-gpu-enricher:latest -f src/gpu-enricher/Dockerfile src/gpu-enricher

push:
	@echo "Pushing images to registry..."
	docker tag ai-finops-gpu-enricher:latest ghcr.io/judeoyovbaire/ai-finops-gpu-enricher:latest
	docker push ghcr.io/judeoyovbaire/ai-finops-gpu-enricher:latest

# Clean targets
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
