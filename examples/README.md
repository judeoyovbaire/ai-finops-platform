# AI FinOps Platform - Integration Examples

This directory contains example Kubernetes manifests for integrating GPU workloads with the AI FinOps Platform.

## Examples

| File | Description | Framework | Use Case |
|------|-------------|-----------|----------|
| `sample-training-job.yaml` | Basic GPU training job | Generic | Simple training workloads |
| `sample-inference-deployment.yaml` | Model serving deployment | KServe | Production inference |
| `pytorch-distributed-training.yaml` | Multi-node distributed training | PyTorch + Kubeflow | Large model training |
| `huggingface-tgi-inference.yaml` | LLM inference server | Hugging Face TGI | Text generation APIs |
| `ray-cluster.yaml` | Distributed ML cluster | Ray | HPO, distributed training, serving |

## Cost Attribution Labels

All examples include the following labels for cost tracking:

```yaml
labels:
  ai-finops.io/team: "team-name"        # Required: Team for cost allocation
  ai-finops.io/project: "project-name"  # Project or application name
  ai-finops.io/model: "model-name"      # ML model being trained/served
  ai-finops.io/environment: "prod"      # Environment: dev, staging, prod

annotations:
  ai-finops.io/cost-center: "ML-001"    # Finance cost center code
  ai-finops.io/budget-owner: "email"    # Budget owner contact
  ai-finops.io/spot-eligible: "true"    # Can use spot instances
```

## Prerequisites

### PyTorch Distributed Training
```bash
# Install Kubeflow Training Operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```

### Hugging Face TGI
```bash
# Create Hugging Face token secret
kubectl create secret generic hf-token \
  --from-literal=token=hf_xxxxx \
  -n ml-inference
```

### Ray Cluster
```bash
# Install KubeRay Operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator
```

## Usage

### Deploy an Example
```bash
# Create namespace
kubectl create namespace ml-workloads

# Deploy PyTorch training job
kubectl apply -f pytorch-distributed-training.yaml

# Check cost attribution
curl http://localhost:8080/api/v1/costs/team/nlp-team
```

### View Costs by Project
```bash
# Get costs for a specific project
curl "http://localhost:8080/api/v1/costs/summary?project=bert-sentiment"
```

### Monitor GPU Utilization
```bash
# Check GPU metrics
curl http://localhost:8080/api/v1/gpu/utilization

# Get recommendations
curl http://localhost:8080/api/v1/recommendations
```

## Best Practices

1. **Always add team labels** - Required for cost allocation
2. **Set resource requests/limits** - Enables accurate cost calculation
3. **Use spot-eligible annotation** - Mark fault-tolerant workloads for savings
4. **Enable Prometheus scraping** - Add `prometheus.io/scrape: "true"` annotation
5. **Use appropriate node selectors** - Match workload to GPU type
