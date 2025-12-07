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
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              AI FinOps Platform                                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────────────┐          │
│  │   NVIDIA    │  │  OpenCost   │  │  Infracost  │  │     GPU Enricher      │          │
│  │    DCGM     │  │   (CNCF)    │  │             │  │                       │          │
│  │  Exporter   │  │             │  │ • Live      │  │ • Cost summary        │          │
│  │             │  │ • K8s costs │  │   pricing   │  │ • Budget forecasting  │          │
│  │ • GPU util  │  │ • Cloud API │  │ • Multi-    │  │ • Anomaly detection   │          │
│  │ • Temp/Mem  │  │ • Namespace │  │   cloud     │  │ • Right-sizing        │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │ • Chargeback reports  │          │
│         │                │                │         └───────────┬───────────┘          │
│         │                │                │                     │                      │
│         └────────────────┴────────────────┴─────────────────────┤                      │
│                                    │                            │                      │
│                                    ▼                            │                      │
│          ┌──────────────────────────────────────────┐           │                      │
│          │              Prometheus v3.8.0           │           │                      │
│          └─────────────────────┬────────────────────┘           │                      │
│                                │                                │                      │
│           ┌────────────────────┼────────────────────┐           │                      │
│           │                    │                    │           │                      │
│           ▼                    ▼                    ▼           │                      │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │  ┌────────────────┐  │
│   │Alertmanager │      │   Grafana   │      │   Thanos    │     │  │ Cloud Billing  │  │
│   │   v0.29.0   │      │   v12.3.0   │      │   v0.40.1   │     │  │      APIs      │  │
│   └──────┬──────┘      └─────────────┘      │             │     │  ├────────────────┤  │
│          │                                  │ • Query     │     │  │ • AWS Cost     │  │
│          ▼                                  │ • Store     │     └─▶│   Explorer     │  │
│   ┌─────────────┐                           │ • Compactor │        │ • GCP Billing  │  │
│   │ Slack/PD/   │                           └──────┬──────┘        │ • Azure Cost   │  │
│   │  Webhooks   │                                  │               └────────────────┘  │
│   └─────────────┘                                  ▼                                   │
│                                         ┌──────────────────┐                           │
│                                         │  Object Storage  │                           │
│                                         │ S3 / GCS / Blob  │                           │
│                                         └──────────────────┘                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
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

### Core Features
- **GPU Utilization Monitoring**: Real-time metrics via NVIDIA DCGM Exporter
- **OpenCost Integration**: CNCF-backed Kubernetes cost allocation
- **Cost-per-Inference Tracking**: Calculate actual cost for each model request
- **Team/Project Attribution**: Label-based cost allocation and chargeback
- **Idle Resource Detection**: Automatic alerts for underutilized GPUs
- **Spot vs On-Demand Analysis**: Recommendations for instance optimization

### Budget & Forecasting (Phase 2)
- **Budget Forecasting**: Projected end-of-month spend with trend analysis
- **Budget Alerts API**: Real-time budget status (on_track, warning, critical, exceeded)
- **Historical Cost Trending**: 7-day, 30-day cost averages and trends
- **Days Until Exhaustion**: Predictive budget exhaustion warnings

### Alerting & Notifications (Phase 2)
- **Alertmanager Integration**: Prometheus alerting with routing rules
- **Slack Notifications**: Channel-based alerts by severity
- **PagerDuty Integration**: Critical alerts with escalation
- **Generic Webhooks**: Custom integration support

### Security (Phase 2)
- **NetworkPolicies**: Zero-trust network isolation between services
- **External Secrets Operator**: AWS Secrets Manager integration
- **Resource Quotas**: Namespace-level resource limits

### Analytics & Intelligence (Phase 3)
- **ML-Based Anomaly Detection**: Statistical analysis using Z-score, IQR, and Moving Average methods to detect cost spikes, utilization anomalies, and efficiency degradation
- **Automated Right-Sizing**: GPU instance recommendations based on utilization patterns (downsize, upsize, terminate, spot migration)
- **AWS Cost Explorer Integration**: Actual billing data from cloud provider for cost comparison and accuracy validation
- **Chargeback Reports**: Automated generation of cost allocation reports in CSV, PDF, and JSON formats

### Multi-Cluster (Phase 3)
- **Thanos Integration**: Long-term metric storage with S3/GCS backend
- **Global Query View**: Unified queries across multiple Kubernetes clusters
- **Metric Deduplication**: Automatic deduplication across Prometheus replicas
- **Downsampling**: Efficient storage with configurable retention (30d raw, 90d 5m, 365d 1h)

### Infrastructure
- **REST API**: Query costs, recommendations, and forecasts programmatically
- **Grafana Dashboards**: Pre-built visualizations for all metrics
- **Multi-cloud Pricing**: Infracost integration for live pricing (AWS, GCP, Azure)

## Architecture

### Components

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| GPU Metrics | NVIDIA DCGM Exporter | 4.4.2 | Collect GPU utilization, memory, temperature |
| K8s Costs | OpenCost | 1.118.0 | Kubernetes resource cost allocation |
| GPU Enricher | Python/Flask | 3.12 | Enrich OpenCost with GPU metrics, provide API |
| Metrics Storage | Prometheus | 3.8.0 | Time-series database for all metrics |
| Alerting | Alertmanager | 0.29.0 | Alert routing to Slack/PagerDuty/Webhooks |
| Visualization | Grafana | 12.3.0 | Dashboards and alerting |
| Pricing | Infracost | - | Multi-cloud GPU pricing API |
| Secrets | External Secrets Operator | - | AWS Secrets Manager integration |
| Multi-Cluster | Thanos | 0.40.1 | Long-term storage, global queries, deduplication |
| Billing | AWS Cost Explorer | - | Actual cloud billing data integration |

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
│       │   ├── thanos.yaml         # Multi-cluster aggregation (Phase 3)
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
│       ├── anomaly.py              # ML-based anomaly detection (Phase 3)
│       ├── rightsizing.py          # Right-sizing recommendations (Phase 3)
│       ├── billing.py              # AWS Cost Explorer integration (Phase 3)
│       ├── reports.py              # Chargeback report generation (Phase 3)
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

**Core Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/api/v1/costs/summary` | GET | Cost summary by team |
| `/api/v1/costs/team/<team>` | GET | Detailed costs for a team |
| `/api/v1/recommendations` | GET | Optimization recommendations |
| `/api/v1/gpu/utilization` | GET | Current GPU utilization |

**Budget & Forecasting Endpoints (Phase 2):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/budget/forecast` | GET | Budget forecast for all teams |
| `/api/v1/budget/forecast/<team>` | GET | Budget forecast for specific team |
| `/api/v1/budget/alerts` | GET | Active budget alerts |
| `/api/v1/trends/costs` | GET | Historical cost trends (7d, 30d, 90d) |

**Analytics & Intelligence Endpoints (Phase 3):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/anomalies` | GET | Detected cost/utilization anomalies |
| `/api/v1/rightsizing` | GET | GPU right-sizing recommendations |
| `/api/v1/billing/actual` | GET | Actual costs from AWS Cost Explorer |
| `/api/v1/billing/comparison` | GET | Estimated vs actual cost comparison |
| `/api/v1/reports/chargeback` | GET | Generate chargeback report (CSV/PDF/JSON) |
| `/api/v1/reports/chargeback/preview` | GET | Preview chargeback report data |

Example:
```bash
# Get cost summary
curl http://localhost:8080/api/v1/costs/summary

# Get recommendations (filter by severity)
curl http://localhost:8080/api/v1/recommendations?severity=high

# Get budget forecast
curl http://localhost:8080/api/v1/budget/forecast

# Get cost trends (30-day period)
curl http://localhost:8080/api/v1/trends/costs?period=30d

# Get detected anomalies (Phase 3)
curl http://localhost:8080/api/v1/anomalies

# Get right-sizing recommendations (Phase 3)
curl http://localhost:8080/api/v1/rightsizing

# Compare estimated vs actual costs (Phase 3)
curl http://localhost:8080/api/v1/billing/comparison

# Generate chargeback report as CSV (Phase 3)
curl "http://localhost:8080/api/v1/reports/chargeback?format=csv&period=monthly"
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

### Phase 2 - Enhanced (Complete)
- [x] Historical cost trending and forecasting (Prometheus recording rules)
- [x] Slack/PagerDuty/Webhook notifications (Alertmanager)
- [x] Multi-cloud pricing API integration (Infracost)
- [x] Budget forecasting API endpoints
- [x] NetworkPolicies for zero-trust security
- [x] External Secrets Operator integration
- [x] Resource quotas and LimitRanges

### Phase 3 - Advanced (Complete)
- [x] ML-based anomaly detection (Z-score, IQR, Moving Average)
- [x] Automated right-sizing recommendations
- [x] Cloud billing API integration (AWS Cost Explorer)
- [x] Chargeback report generation (PDF/CSV/JSON)
- [x] Multi-cluster aggregation (Thanos)
- [ ] CI/CD pipeline enablement

### Future Enhancements
- [ ] GCP Cloud Billing API integration
- [ ] Azure Cost Management API integration
- [ ] Custom ML models for cost prediction
- [ ] Kubernetes Cost Allocation API (native)

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
