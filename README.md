# AI Infrastructure FinOps Platform

Cost optimization platform for AI/ML workloads providing visibility into GPU utilization, inference costs per model, and actionable recommendations for reducing cloud spend.

[![CI](https://github.com/judeoyovbaire/ai-finops-platform/actions/workflows/ci.yaml/badge.svg)](https://github.com/judeoyovbaire/ai-finops-platform/actions/workflows/ci.yaml)

## The Problem

Organizations running AI/ML workloads face significant cost challenges:

| Challenge | Impact |
|-----------|--------|
| No GPU utilization visibility | 40-60% of GPU capacity sits idle |
| Unknown cost per inference | Can't optimize high-cost models |
| No team/project attribution | No accountability for spend |
| Reactive cost management | Budget overruns discovered too late |
| Spot instance complexity | Missing savings opportunities |

## The Solution

This platform provides comprehensive cost observability for AI infrastructure by combining [OpenCost](https://opencost.io/) for Kubernetes cost allocation with GPU-specific metrics from NVIDIA DCGM:

```
┌────────────────────────────────────────────────────────────────┐
│                    AI FinOps Platform                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │   NVIDIA    │  │  OpenCost   │  │     GPU Enricher       ││
│  │    DCGM     │  │   (CNCF)    │  │                         ││
│  │  Exporter   │  │             │  │  • Joins GPU + K8s     ││
│  │             │  │ • K8s costs │  │  • Idle detection      ││
│  │ • GPU util  │  │ • Cloud API │  │  • Spot recommendations││
│  │ • Temp/Mem  │  │ • Namespace │  │  • REST API            ││
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘│
│         │                │                     │               │
│         └────────────────┼─────────────────────┘               │
│                          ▼                                     │
│                   ┌─────────────┐                              │
│                   │ Prometheus  │                              │
│                   │   v3.0.1    │                              │
│                   └──────┬──────┘                              │
│                          ▼                                     │
│                   ┌─────────────┐                              │
│                   │   Grafana   │                              │
│                   │   v11.4.0   │                              │
│                   └─────────────┘                              │
└────────────────────────────────────────────────────────────────┘
```

## Key Outcomes

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| GPU utilization visibility | None | Real-time | Identify idle resources |
| Cost attribution | Manual estimates | Per-model tracking | Enable chargeback |
| Idle GPU detection | Days to discover | Instant alerts | Reduce waste |
| Spot recommendations | Ad-hoc | Automated | 60-70% savings potential |
| Budget tracking | Monthly surprise | Daily visibility | Proactive management |

## Features

- **GPU Utilization Monitoring**: Real-time metrics via NVIDIA DCGM Exporter
- **OpenCost Integration**: CNCF-backed Kubernetes cost allocation
- **Cost-per-Inference Tracking**: Calculate actual cost for each model request
- **Team/Project Attribution**: Label-based cost allocation and chargeback
- **Idle Resource Detection**: Automatic alerts for underutilized GPUs
- **Spot vs On-Demand Analysis**: Recommendations for instance optimization
- **Budget Alerting**: Proactive notifications before budget breaches
- **REST API**: Query costs and recommendations programmatically
- **Grafana Dashboards**: Pre-built visualizations for all metrics

## Architecture

### Components

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| GPU Metrics | NVIDIA DCGM Exporter | 3.3.9 | Collect GPU utilization, memory, temperature |
| K8s Costs | OpenCost | 1.113.0 | Kubernetes resource cost allocation |
| GPU Enricher | Python/Flask | 3.12 | Enrich OpenCost with GPU metrics, provide API |
| Metrics Storage | Prometheus | 3.0.1 | Time-series database for all metrics |
| Visualization | Grafana | 11.4.0 | Dashboards and alerting |

### Metrics Collected

**GPU Metrics (via DCGM):**
- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization percentage
- `DCGM_FI_DEV_MEM_COPY_UTIL` - Memory utilization
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature
- `DCGM_FI_DEV_POWER_USAGE` - Power consumption

**Cost Metrics (GPU Enricher):**
- `ai_finops_gpu_cost_per_hour` - Hourly cost by GPU and team
- `ai_finops_team_cost_daily` - Daily cost by team (GPU + K8s)
- `ai_finops_gpu_cost_daily` - Daily GPU-only cost
- `ai_finops_idle_gpu_hours` - Wasted GPU hours
- `ai_finops_spot_savings_potential` - Potential spot savings
- `ai_finops_gpu_efficiency` - GPU efficiency score

**Operational Metrics:**
- `ai_finops_prometheus_query_errors_total` - Query error tracking
- `ai_finops_opencost_query_errors_total` - OpenCost API errors
- `ai_finops_query_duration_seconds` - Query latency histogram

### Label Strategy

Cost attribution uses Kubernetes labels:

```yaml
metadata:
  labels:
    ai-finops.io/team: "ml-platform"
    ai-finops.io/project: "recommendation-engine"
    ai-finops.io/model: "product-embeddings"
    ai-finops.io/environment: "production"
  annotations:
    ai-finops.io/cost-center: "ML-001"
    ai-finops.io/budget-owner: "ml-platform@company.com"
```

## Project Structure

```
ai-finops-platform/
├── .github/
│   └── workflows/
│       └── ci.yaml                 # CI pipeline (lint, test, build, scan)
├── deploy/
│   └── kubernetes/
│       ├── base/                   # Base Kustomize manifests
│       │   ├── namespace.yaml
│       │   ├── dcgm-exporter.yaml  # NVIDIA GPU metrics
│       │   ├── opencost.yaml       # OpenCost deployment
│       │   ├── gpu-enricher.yaml   # GPU cost enrichment service
│       │   ├── prometheus.yaml     # Prometheus with alert rules
│       │   ├── grafana.yaml        # Grafana with datasources
│       │   ├── grafana-dashboards.yaml
│       │   └── kustomization.yaml
│       └── overlays/
│           ├── local/              # Local/Kind with mock metrics
│           │   ├── mock-gpu-metrics.yaml
│           │   ├── dcgm-disable-patch.yaml
│           │   └── kustomization.yaml
│           └── aws/                # AWS EKS with persistent storage
│               ├── prometheus-storage-patch.yaml
│               ├── grafana-storage-patch.yaml
│               └── kustomization.yaml
├── src/
│   └── gpu-enricher/               # GPU cost enrichment service
│       ├── main.py                 # Flask application
│       ├── test_main.py            # Unit tests (50+ tests)
│       ├── requirements.txt
│       └── Dockerfile
├── dashboards/
│   └── gpu-overview.json           # Comprehensive GPU dashboard
├── examples/
│   ├── sample-training-job.yaml    # Training job with labels
│   └── sample-inference-deployment.yaml  # Inference with HPA
├── scripts/
│   └── deploy-local.sh             # Local deployment automation
├── Makefile                        # Build and deployment commands
├── README.md
└── LICENSE
```

## Getting Started

### Prerequisites

- Kubernetes cluster (1.21+)
- kubectl configured
- kustomize installed
- For GPU metrics: NVIDIA GPU Operator or drivers

### Quick Start - Local (Kind)

```bash
# Clone the repository
git clone https://github.com/judeoyovbaire/ai-finops-platform.git
cd ai-finops-platform

# Create Kind cluster (if needed)
kind create cluster --name ai-finops

# Deploy to local cluster (uses mock GPU metrics)
make deploy-local

# Access services (run in separate terminals)
make port-forward-grafana     # Grafana at localhost:3000 (admin/admin)
make port-forward-prometheus  # Prometheus at localhost:9090
make port-forward-opencost    # OpenCost UI at localhost:9091
make port-forward-enricher    # GPU Enricher API at localhost:8080
```

### Quick Start - AWS EKS

```bash
# Deploy to EKS cluster with GPU nodes
make deploy-aws

# Verify deployment
make status
```

### API Endpoints

The GPU Enricher provides a REST API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/costs/summary` | GET | Cost summary by team |
| `/api/v1/costs/team/<team>` | GET | Detailed costs for a team |
| `/api/v1/recommendations` | GET | Optimization recommendations |
| `/api/v1/gpu/utilization` | GET | Current GPU utilization |

Example:
```bash
# Get cost summary
curl http://localhost:8080/api/v1/costs/summary

# Get recommendations (filter by severity)
curl http://localhost:8080/api/v1/recommendations?severity=high
```

### Makefile Commands

```bash
make help                       # Show all commands

# Deployment
make deploy-local               # Deploy to local Kind cluster
make deploy-aws                 # Deploy to AWS EKS
make destroy                    # Remove all resources
make status                     # Check deployment status

# Port Forwarding
make port-forward-grafana       # Grafana at localhost:3000
make port-forward-prometheus    # Prometheus at localhost:9090
make port-forward-opencost      # OpenCost UI at localhost:9091
make port-forward-enricher      # GPU Enricher API at localhost:8080

# Development
make test                       # Run unit tests
make test-cov                   # Run tests with coverage
make lint                       # Lint Python code
make lint-fix                   # Auto-fix lint issues
make build                      # Build container image
make push                       # Push to registry
make clean                      # Clean build artifacts
```

## Dashboards

### GPU Overview Dashboard

Comprehensive dashboard with 15 panels across 4 sections:

**Cost Overview:**
- Total daily cost (GPU + K8s)
- GPU-only daily cost
- Potential spot savings
- Idle GPU count

**GPU Utilization:**
- Utilization by node (timeseries)
- Average utilization by team (gauge)
- Memory utilization
- Temperature monitoring

**Efficiency & Recommendations:**
- GPU efficiency by team
- Idle GPU hours by team
- Spot savings potential by node

**Service Health:**
- Query error rates
- Query latency (p50, p95)

## Configuration

### Cloud Pricing

Configure GPU pricing in `deploy/kubernetes/base/gpu-enricher.yaml`:

```yaml
pricing:
  aws:
    g4dn.xlarge:
      on_demand: 0.526
      spot_avg: 0.158
    g5.xlarge:
      on_demand: 1.006
      spot_avg: 0.302
    p3.2xlarge:
      on_demand: 3.06
      spot_avg: 0.918
  gcp:
    n1-standard-4-t4:
      on_demand: 0.35
      spot_avg: 0.11
  azure:
    Standard_NC4as_T4_v3:
      on_demand: 0.526
      spot_avg: 0.158
```

### Alert Thresholds

Alerts are configured in `deploy/kubernetes/base/prometheus.yaml`:

```yaml
groups:
  - name: ai-finops-alerts
    rules:
      - alert: GPUIdleHigh
        expr: avg by (node, gpu) (DCGM_FI_DEV_GPU_UTIL) < 20
        for: 30m
        labels:
          severity: warning

      - alert: DailyBudgetExceeded
        expr: sum(ai_finops_team_cost_daily) > 1000
        labels:
          severity: critical

      - alert: GPUTemperatureHigh
        expr: DCGM_FI_DEV_GPU_TEMP > 85
        for: 5m
        labels:
          severity: warning
```

## Roadmap

### Phase 1 - MVP (Complete)
- [x] Project structure and documentation
- [x] NVIDIA DCGM Exporter deployment
- [x] OpenCost integration (CNCF)
- [x] GPU Enricher service with REST API
- [x] Prometheus configuration with alerts
- [x] Comprehensive Grafana dashboard
- [x] Idle GPU detection and alerting
- [x] Unit tests (50+ test cases)
- [x] Local development with mock metrics

### Phase 2 - Enhanced
- [ ] Spot vs on-demand automated recommendations
- [ ] Historical cost trending and forecasting
- [ ] Slack/email/PagerDuty notifications
- [ ] Multi-cloud pricing API integration
- [ ] Budget forecasting with ML

### Phase 3 - Advanced
- [ ] ML-based anomaly detection
- [ ] Automated right-sizing recommendations
- [ ] Cloud billing API integration (actual costs)
- [ ] Chargeback report generation (PDF/CSV)
- [ ] Multi-cluster aggregation (Thanos/Cortex)

## Integration with MLOps Platform

This platform integrates with ML serving frameworks:

```yaml
# KServe InferenceService with cost labels
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: recommendation-model
  labels:
    ai-finops.io/team: "ml-platform"
    ai-finops.io/model: "recommendation-v2"
    ai-finops.io/environment: "production"
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      resources:
        limits:
          nvidia.com/gpu: "1"
```

## Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Lint code
make lint
```

## Contributing

Contributions welcome! Please read the contributing guidelines first.

## License

MIT License - see LICENSE for details.

## Author

**Jude Oyovbaire** - Senior DevOps Engineer & Platform Architect

- Website: [judaire.io](https://judaire.io)
- LinkedIn: [linkedin.com/in/judeoyovbaire](https://linkedin.com/in/judeoyovbaire)
- GitHub: [github.com/judeoyovbaire](https://github.com/judeoyovbaire)
