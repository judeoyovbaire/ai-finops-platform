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

This platform provides comprehensive cost observability for AI infrastructure:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI FinOps Platform                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   NVIDIA    │    │    Cost     │    │       Grafana           │ │
│  │    DCGM     │───▶│  Calculator │───▶│      Dashboards         │ │
│  │  Exporter   │    │   Service   │    │                         │ │
│  └─────────────┘    └─────────────┘    │  - GPU Utilization      │ │
│         │                  │           │  - Cost per Model       │ │
│         │                  │           │  - Team Chargeback      │ │
│         ▼                  ▼           │  - Idle Detection       │ │
│  ┌─────────────────────────────────┐   │  - Recommendations      │ │
│  │          Prometheus             │   └─────────────────────────┘ │
│  │    (Metrics Storage)            │                               │
│  └─────────────────────────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
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
- **Cost-per-Inference Tracking**: Calculate actual cost for each model request
- **Team/Project Attribution**: Label-based cost allocation and chargeback
- **Idle Resource Detection**: Automatic alerts for underutilized GPUs
- **Spot vs On-Demand Analysis**: Recommendations for instance optimization
- **Budget Alerting**: Proactive notifications before budget breaches
- **Historical Trending**: Track cost patterns over time
- **Grafana Dashboards**: Pre-built visualizations for all metrics

## Architecture

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| GPU Metrics | NVIDIA DCGM Exporter | Collect GPU utilization, memory, temperature |
| Cost Calculator | Python Service | Join metrics with pricing, calculate costs |
| Metrics Storage | Prometheus | Time-series database for all metrics |
| Visualization | Grafana | Dashboards and alerting |
| Alert Manager | Prometheus Alertmanager | Notifications (Slack, email, PagerDuty) |

### Metrics Collected

**GPU Metrics (via DCGM):**
- `DCGM_FI_DEV_GPU_UTIL` - GPU utilization percentage
- `DCGM_FI_DEV_MEM_COPY_UTIL` - Memory utilization
- `DCGM_FI_DEV_GPU_TEMP` - GPU temperature
- `DCGM_FI_DEV_POWER_USAGE` - Power consumption

**Cost Metrics (calculated):**
- `ai_finops_gpu_cost_per_hour` - Hourly cost by GPU type
- `ai_finops_inference_cost_total` - Cost per inference request
- `ai_finops_idle_gpu_hours` - Wasted GPU hours
- `ai_finops_team_cost_daily` - Daily cost by team label

### Label Strategy

Cost attribution uses Kubernetes labels:

```yaml
labels:
  ai-finops.io/team: "ml-platform"
  ai-finops.io/project: "recommendation-engine"
  ai-finops.io/model: "product-embeddings"
  ai-finops.io/environment: "production"
```

## Project Structure

```
ai-finops-platform/
├── .github/
│   └── workflows/
│       └── ci.yaml              # CI pipeline
├── deploy/
│   └── kubernetes/
│       ├── base/                # Base Kustomize manifests
│       │   ├── namespace.yaml
│       │   ├── dcgm-exporter.yaml
│       │   ├── prometheus.yaml
│       │   ├── grafana.yaml
│       │   ├── cost-calculator.yaml
│       │   └── kustomization.yaml
│       └── overlays/
│           ├── local/           # Local/Kind deployment
│           └── aws/             # AWS EKS deployment
├── src/
│   ├── collector/               # Custom metrics collector (optional)
│   ├── calculator/              # Cost calculation service
│   │   ├── main.py
│   │   ├── pricing.py
│   │   ├── metrics.py
│   │   └── requirements.txt
│   └── api/                     # REST API for cost queries
├── dashboards/
│   ├── gpu-utilization.json     # GPU metrics dashboard
│   ├── cost-attribution.json    # Cost by team/model dashboard
│   └── recommendations.json     # Optimization recommendations
├── docs/
│   └── architecture.md          # Detailed architecture docs
├── scripts/
│   ├── deploy-local.sh          # Local deployment script
│   └── generate-test-data.sh    # Generate sample metrics
├── examples/
│   └── sample-workload.yaml     # Example GPU workload with labels
├── Makefile
├── README.md
└── LICENSE
```

## Getting Started

### Prerequisites

- Kubernetes cluster with GPU nodes
- NVIDIA GPU Operator installed (or NVIDIA drivers)
- kubectl configured
- Helm 3.x (optional, for Prometheus stack)

### Quick Start - Local (Kind)

```bash
# Clone the repository
git clone https://github.com/judeoyovbaire/ai-finops-platform.git
cd ai-finops-platform

# Deploy to local Kind cluster (simulated GPU metrics)
make deploy-local

# Access dashboards
make port-forward-grafana  # Grafana at localhost:3000
make port-forward-prometheus  # Prometheus at localhost:9090
```

### Quick Start - AWS EKS

```bash
# Deploy to EKS cluster with GPU nodes
make deploy-aws

# Verify deployment
make status
```

### Using the Makefile

```bash
make help                    # Show all commands

# Deployment
make deploy-local            # Deploy to local Kind cluster
make deploy-aws              # Deploy to AWS EKS
make destroy                 # Remove all resources

# Development
make port-forward-grafana    # Forward Grafana to localhost:3000
make port-forward-prometheus # Forward Prometheus to localhost:9090
make test                    # Run tests
make lint                    # Lint Python code
```

## Dashboards

### GPU Utilization Overview

Shows real-time GPU utilization across all nodes:
- Utilization heatmap by node
- Memory usage trends
- Idle GPU detection
- Temperature monitoring

### Cost Attribution

Break down costs by dimensions:
- Cost per team (daily/weekly/monthly)
- Cost per model
- Cost per environment
- Trend analysis

### Optimization Recommendations

Actionable insights:
- Underutilized GPUs (candidates for scale-down)
- High-cost models (optimization targets)
- Spot instance opportunities
- Right-sizing suggestions

## Configuration

### Cloud Pricing

Configure GPU instance pricing in `src/calculator/pricing.py`:

```python
GPU_PRICING = {
    "aws": {
        "g4dn.xlarge": {"on_demand": 0.526, "spot_avg": 0.158},
        "g4dn.2xlarge": {"on_demand": 0.752, "spot_avg": 0.226},
        "g5.xlarge": {"on_demand": 1.006, "spot_avg": 0.302},
        "p3.2xlarge": {"on_demand": 3.06, "spot_avg": 0.918},
    },
    "gcp": {
        "n1-standard-4-t4": {"on_demand": 0.35, "spot_avg": 0.11},
        "a2-highgpu-1g": {"on_demand": 3.67, "spot_avg": 1.10},
    }
}
```

### Alert Thresholds

Configure alerts in `deploy/kubernetes/base/prometheus-rules.yaml`:

```yaml
groups:
  - name: ai-finops-alerts
    rules:
      - alert: GPUIdleHigh
        expr: avg(DCGM_FI_DEV_GPU_UTIL) by (node) < 20
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "GPU idle for 30+ minutes"

      - alert: DailyBudgetExceeded
        expr: sum(ai_finops_team_cost_daily) > 1000
        labels:
          severity: critical
        annotations:
          summary: "Daily GPU budget exceeded $1000"
```

## Roadmap

### Phase 1 - MVP (Current)
- [x] Project structure and documentation
- [ ] DCGM Exporter deployment
- [ ] Prometheus configuration
- [ ] Basic Grafana dashboards
- [ ] Cost calculator service
- [ ] Idle GPU alerting

### Phase 2 - Enhanced
- [ ] Spot vs on-demand recommendations
- [ ] Historical cost trending
- [ ] Budget forecasting
- [ ] Slack/email notifications
- [ ] REST API for cost queries

### Phase 3 - Advanced
- [ ] ML-based anomaly detection
- [ ] Automated right-sizing recommendations
- [ ] Cloud billing API integration (actual costs)
- [ ] Chargeback report generation (PDF/CSV)
- [ ] Multi-cluster aggregation

## Integration with MLOps Platform

This platform integrates seamlessly with the [MLOps Platform](https://github.com/judeoyovbaire/mlops-platform):

```yaml
# Add labels to KServe InferenceServices
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-model
  labels:
    ai-finops.io/team: "ml-platform"
    ai-finops.io/model: "my-model"
spec:
  predictor:
    # ... predictor config
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