#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo -e "  AI FinOps Platform - Local Deployment  "
echo -e "========================================${NC}"

# Check prerequisites
echo -e "\n${BLUE}Checking prerequisites...${NC}"

if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}[✗] kubectl not found. Please install kubectl.${NC}"
    exit 1
fi
echo -e "${GREEN}[✓]${NC} kubectl is installed"

if ! command -v kustomize &> /dev/null; then
    echo -e "${RED}[✗] kustomize not found. Please install kustomize.${NC}"
    exit 1
fi
echo -e "${GREEN}[✓]${NC} kustomize is installed"

# Check for Kind cluster
if kubectl cluster-info &> /dev/null; then
    echo -e "${GREEN}[✓]${NC} Kubernetes cluster is accessible"
else
    echo -e "${RED}[✗] No Kubernetes cluster found. Please create a Kind cluster first.${NC}"
    echo -e "${BLUE}[i]${NC} Run: kind create cluster --name ai-finops"
    exit 1
fi

# Deploy using Kustomize
echo -e "\n${BLUE}Deploying AI FinOps Platform...${NC}"

# Apply the local overlay
kubectl apply -k deploy/kubernetes/overlays/local

echo -e "${GREEN}[✓]${NC} Manifests applied"

# Wait for deployments
echo -e "\n${BLUE}Waiting for deployments to be ready...${NC}"

echo -e "${BLUE}[i]${NC} Waiting for Prometheus..."
kubectl rollout status deployment/prometheus -n ai-finops --timeout=120s

echo -e "${BLUE}[i]${NC} Waiting for OpenCost..."
kubectl rollout status deployment/opencost -n ai-finops --timeout=120s

echo -e "${BLUE}[i]${NC} Waiting for GPU Enricher..."
kubectl rollout status deployment/gpu-enricher -n ai-finops --timeout=120s

echo -e "${BLUE}[i]${NC} Waiting for Grafana..."
kubectl rollout status deployment/grafana -n ai-finops --timeout=120s

echo -e "${BLUE}[i]${NC} Waiting for Mock GPU Metrics..."
kubectl rollout status deployment/mock-gpu-metrics -n ai-finops --timeout=60s

echo -e "${GREEN}[✓]${NC} All deployments ready"

# Print access information
echo -e "\n${BLUE}========================================"
echo -e "${GREEN}  Deployment Complete!                  "
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Components Deployed:${NC}"
echo -e "  • Prometheus      - Metrics collection & alerting"
echo -e "  • OpenCost        - Kubernetes cost allocation"
echo -e "  • GPU Enricher    - GPU cost enrichment & recommendations"
echo -e "  • Grafana         - Visualization dashboards"
echo -e "  • Mock GPU Metrics- Simulated GPU workloads"

echo -e "\n${BLUE}Access URLs:${NC}"
echo -e "  Grafana:      ${GREEN}http://localhost:3000${NC} (admin/admin)"
echo -e "  Prometheus:   ${GREEN}http://localhost:9090${NC}"
echo -e "  OpenCost UI:  ${GREEN}http://localhost:9091${NC}"
echo -e "  GPU Enricher: ${GREEN}http://localhost:8080${NC}"

echo -e "\n${BLUE}Port Forward Commands:${NC}"
echo -e "  make port-forward-grafana"
echo -e "  make port-forward-prometheus"
echo -e "  make port-forward-opencost"
echo -e "  make port-forward-enricher"

echo -e "\n${BLUE}API Endpoints (GPU Enricher):${NC}"
echo -e "  GET /api/v1/costs/summary      - Cost summary by team"
echo -e "  GET /api/v1/recommendations    - Optimization recommendations"
echo -e "  GET /api/v1/gpu/utilization    - GPU utilization metrics"

echo -e "\n${BLUE}Useful Commands:${NC}"
echo -e "  kubectl get pods -n ai-finops     # View pods"
echo -e "  make status                        # Check deployment status"
echo -e "  make destroy                       # Remove deployment"