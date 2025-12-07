"""
AI FinOps GPU Enricher Service

Enriches OpenCost Kubernetes cost data with GPU-specific metrics from DCGM.
Provides cost attribution, idle detection, and optimization recommendations.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
import yaml
from flask import Flask, jsonify, request
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =============================================================================
# Prometheus Metrics
# =============================================================================

# GPU cost metrics
gpu_cost_hourly = Gauge(
    "ai_finops_gpu_cost_per_hour",
    "Estimated hourly GPU cost in USD",
    ["node", "instance_type", "team", "cloud_provider"],
)

team_cost_daily = Gauge(
    "ai_finops_team_cost_daily",
    "Daily cost by team in USD (GPU + Kubernetes)",
    ["team"],
)

gpu_cost_daily = Gauge(
    "ai_finops_gpu_cost_daily",
    "Daily GPU-only cost by team in USD",
    ["team"],
)

idle_gpu_hours = Gauge(
    "ai_finops_idle_gpu_hours",
    "Hours of idle GPU time (utilization < 20%)",
    ["node", "team"],
)

spot_savings_potential = Gauge(
    "ai_finops_spot_savings_potential",
    "Potential daily savings from using spot instances in USD",
    ["node", "instance_type"],
)

gpu_efficiency = Gauge(
    "ai_finops_gpu_efficiency",
    "GPU efficiency score (0-100)",
    ["node", "team"],
)

# OpenCost integration metrics
opencost_allocation = Gauge(
    "ai_finops_opencost_cost_daily",
    "Daily Kubernetes cost from OpenCost",
    ["namespace", "controller", "pod"],
)

# Operational metrics
prometheus_query_errors = Counter(
    "ai_finops_prometheus_query_errors_total",
    "Total Prometheus query errors",
    ["query_type"],
)

opencost_query_errors = Counter(
    "ai_finops_opencost_query_errors_total",
    "Total OpenCost API query errors",
    ["endpoint"],
)

query_duration = Histogram(
    "ai_finops_query_duration_seconds",
    "Duration of metric queries",
    ["source"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GPUPricing:
    """GPU instance pricing configuration."""

    on_demand: float
    spot_avg: float

    @property
    def spot_savings_pct(self) -> float:
        """Calculate spot savings percentage."""
        if self.on_demand == 0:
            return 0.0
        return (self.on_demand - self.spot_avg) / self.on_demand * 100


@dataclass
class GPUMetrics:
    """GPU metrics for a single GPU."""

    node: str
    gpu_id: str
    utilization: float
    memory_util: float
    temperature: float
    power_usage: float
    team: str = "unknown"
    model: str = "unknown"
    instance_type: str = "unknown"


@dataclass
class TeamCostSummary:
    """Cost summary for a team."""

    team: str
    gpu_cost_daily: float = 0.0
    k8s_cost_daily: float = 0.0
    idle_gpu_hours: float = 0.0
    gpu_count: int = 0
    avg_utilization: float = 0.0
    recommendations: list = field(default_factory=list)

    @property
    def total_cost_daily(self) -> float:
        return self.gpu_cost_daily + self.k8s_cost_daily


@dataclass
class Recommendation:
    """Cost optimization recommendation."""

    type: str
    severity: str  # low, medium, high
    title: str
    description: str
    potential_savings_daily: float
    affected_resources: list


# =============================================================================
# GPU Enricher Service
# =============================================================================


class GPUEnricher:
    """Enriches OpenCost data with GPU metrics and recommendations."""

    # Idle threshold (percentage)
    IDLE_THRESHOLD = 20.0

    def __init__(self, config_path: str, prometheus_url: str, opencost_url: str):
        self.prometheus_url = prometheus_url
        self.opencost_url = opencost_url
        self.pricing = self._load_pricing(config_path)
        self._last_update = None
        self._cached_summary: dict[str, TeamCostSummary] = {}
        logger.info(f"GPU Enricher initialized with {len(self.pricing)} instance types")

    def _load_pricing(self, config_path: str) -> dict[str, GPUPricing]:
        """Load pricing configuration from YAML file."""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            pricing = {}
            for cloud, instances in config.get("pricing", {}).items():
                for instance_type, prices in instances.items():
                    key = f"{cloud}:{instance_type}"
                    pricing[key] = GPUPricing(
                        on_demand=prices["on_demand"],
                        spot_avg=prices["spot_avg"],
                    )
            return pricing
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._default_pricing()
        except Exception as e:
            logger.error(f"Failed to load pricing config: {e}")
            return self._default_pricing()

    def _default_pricing(self) -> dict[str, GPUPricing]:
        """Return default GPU pricing."""
        return {
            "aws:g4dn.xlarge": GPUPricing(on_demand=0.526, spot_avg=0.158),
            "aws:g4dn.2xlarge": GPUPricing(on_demand=0.752, spot_avg=0.226),
            "aws:g5.xlarge": GPUPricing(on_demand=1.006, spot_avg=0.302),
            "aws:p3.2xlarge": GPUPricing(on_demand=3.06, spot_avg=0.918),
        }

    def query_prometheus(self, query: str, query_type: str = "gpu") -> list[dict]:
        """Query Prometheus for metrics with error tracking."""
        start_time = time.time()
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            duration = time.time() - start_time
            query_duration.labels(source="prometheus").observe(duration)
            return result.get("data", {}).get("result", [])
        except requests.exceptions.Timeout:
            logger.error(f"Prometheus query timeout: {query}")
            prometheus_query_errors.labels(query_type=query_type).inc()
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Prometheus query failed: {e}")
            prometheus_query_errors.labels(query_type=query_type).inc()
            return []
        except Exception as e:
            logger.error(f"Unexpected error querying Prometheus: {e}")
            prometheus_query_errors.labels(query_type=query_type).inc()
            return []

    def query_opencost(self, window: str = "1d") -> dict:
        """Query OpenCost allocation API."""
        start_time = time.time()
        try:
            response = requests.get(
                f"{self.opencost_url}/allocation/compute",
                params={
                    "window": window,
                    "aggregate": "namespace,controller",
                    "accumulate": "true",
                },
                timeout=15,
            )
            response.raise_for_status()
            duration = time.time() - start_time
            query_duration.labels(source="opencost").observe(duration)
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("OpenCost query timeout")
            opencost_query_errors.labels(endpoint="allocation").inc()
            return {}
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenCost query failed: {e}")
            opencost_query_errors.labels(endpoint="allocation").inc()
            return {}
        except Exception as e:
            logger.error(f"Unexpected error querying OpenCost: {e}")
            opencost_query_errors.labels(endpoint="allocation").inc()
            return {}

    def get_gpu_metrics(self) -> list[GPUMetrics]:
        """Fetch current GPU metrics from Prometheus."""
        metrics = []

        # Query GPU utilization
        gpu_util = self.query_prometheus("DCGM_FI_DEV_GPU_UTIL", "utilization")

        for metric in gpu_util:
            labels = metric.get("metric", {})
            value = float(metric.get("value", [0, 0])[1])

            gpu_metrics = GPUMetrics(
                node=labels.get("node", "unknown"),
                gpu_id=labels.get("gpu", "0"),
                utilization=value,
                memory_util=0.0,
                temperature=0.0,
                power_usage=0.0,
                team=labels.get("team", "unknown"),
                model=labels.get("model", "unknown"),
                instance_type=labels.get("instance_type", "g4dn.xlarge"),
            )
            metrics.append(gpu_metrics)

        # Enrich with memory utilization
        mem_util = self.query_prometheus("DCGM_FI_DEV_MEM_COPY_UTIL", "memory")
        mem_by_key = {
            (m["metric"].get("node"), m["metric"].get("gpu")): float(
                m.get("value", [0, 0])[1]
            )
            for m in mem_util
        }
        for gpu in metrics:
            gpu.memory_util = mem_by_key.get((gpu.node, gpu.gpu_id), 0.0)

        # Enrich with temperature
        temp = self.query_prometheus("DCGM_FI_DEV_GPU_TEMP", "temperature")
        temp_by_key = {
            (m["metric"].get("node"), m["metric"].get("gpu")): float(
                m.get("value", [0, 0])[1]
            )
            for m in temp
        }
        for gpu in metrics:
            gpu.temperature = temp_by_key.get((gpu.node, gpu.gpu_id), 0.0)

        return metrics

    def get_opencost_allocation(self) -> dict[str, float]:
        """Get cost allocation from OpenCost by namespace."""
        allocations = self.query_opencost("1d")
        costs_by_namespace = {}

        data = allocations.get("data", [])
        if not data:
            return costs_by_namespace

        # OpenCost returns array of allocations
        for alloc_set in data:
            if isinstance(alloc_set, dict):
                for key, alloc in alloc_set.items():
                    if isinstance(alloc, dict):
                        namespace = alloc.get("properties", {}).get(
                            "namespace", "unknown"
                        )
                        total_cost = alloc.get("totalCost", 0.0)
                        if namespace not in costs_by_namespace:
                            costs_by_namespace[namespace] = 0.0
                        costs_by_namespace[namespace] += total_cost

        return costs_by_namespace

    def calculate_costs(self) -> dict[str, TeamCostSummary]:
        """Calculate comprehensive costs combining GPU and OpenCost data."""
        team_summaries: dict[str, TeamCostSummary] = {}
        gpu_metrics = self.get_gpu_metrics()
        opencost_data = self.get_opencost_allocation()

        # Process GPU metrics
        for gpu in gpu_metrics:
            team = gpu.team
            if team not in team_summaries:
                team_summaries[team] = TeamCostSummary(team=team)

            summary = team_summaries[team]
            summary.gpu_count += 1

            # Get pricing
            pricing_key = f"aws:{gpu.instance_type}"
            pricing = self.pricing.get(pricing_key)
            if not pricing:
                pricing = self.pricing.get("aws:g4dn.xlarge", GPUPricing(0.526, 0.158))

            # Calculate hourly cost based on utilization
            utilization_factor = gpu.utilization / 100.0
            hourly_cost = pricing.on_demand * max(utilization_factor, 0.1)

            # Update metrics
            gpu_cost_hourly.labels(
                node=gpu.node,
                instance_type=gpu.instance_type,
                team=team,
                cloud_provider="aws",
            ).set(hourly_cost)

            # Accumulate daily costs (hourly * 24)
            summary.gpu_cost_daily += hourly_cost * 24
            summary.avg_utilization = (
                (summary.avg_utilization * (summary.gpu_count - 1)) + gpu.utilization
            ) / summary.gpu_count

            # Track idle GPUs
            if gpu.utilization < self.IDLE_THRESHOLD:
                summary.idle_gpu_hours += 1
                idle_gpu_hours.labels(node=gpu.node, team=team).set(1)

            # Calculate spot savings potential
            potential_savings = (pricing.on_demand - pricing.spot_avg) * 24
            spot_savings_potential.labels(
                node=gpu.node, instance_type=gpu.instance_type
            ).set(potential_savings)

            # GPU efficiency
            efficiency = min(100, gpu.utilization * 1.2)  # Slight boost for active GPUs
            gpu_efficiency.labels(node=gpu.node, team=team).set(efficiency)

        # Add OpenCost Kubernetes costs
        for namespace, k8s_cost in opencost_data.items():
            # Map namespace to team (could be enhanced with label mapping)
            team = namespace
            if team not in team_summaries:
                team_summaries[team] = TeamCostSummary(team=team)
            team_summaries[team].k8s_cost_daily += k8s_cost

            opencost_allocation.labels(
                namespace=namespace, controller="", pod=""
            ).set(k8s_cost)

        # Update team cost metrics
        for team, summary in team_summaries.items():
            team_cost_daily.labels(team=team).set(summary.total_cost_daily)
            gpu_cost_daily.labels(team=team).set(summary.gpu_cost_daily)

        # Generate recommendations
        self._generate_recommendations(team_summaries)

        self._cached_summary = team_summaries
        self._last_update = datetime.now(timezone.utc)

        logger.info(
            f"Updated costs for {len(team_summaries)} teams, "
            f"{len(gpu_metrics)} GPUs"
        )
        return team_summaries

    def _generate_recommendations(
        self, summaries: dict[str, TeamCostSummary]
    ) -> None:
        """Generate optimization recommendations for each team."""
        for team, summary in summaries.items():
            recommendations = []

            # Idle GPU recommendation
            if summary.idle_gpu_hours > 0:
                idle_cost = (summary.gpu_cost_daily / max(summary.gpu_count, 1)) * (
                    summary.idle_gpu_hours / 24
                )
                recommendations.append(
                    Recommendation(
                        type="idle_gpu",
                        severity="high" if summary.idle_gpu_hours > 4 else "medium",
                        title="Scale down idle GPUs",
                        description=f"{summary.idle_gpu_hours:.0f} GPU hours with <20% utilization",
                        potential_savings_daily=idle_cost,
                        affected_resources=[team],
                    )
                )

            # Spot instance recommendation
            if summary.gpu_count > 0 and summary.avg_utilization < 80:
                # Non-critical workloads could use spot
                spot_savings = summary.gpu_cost_daily * 0.6  # ~60% savings on spot
                recommendations.append(
                    Recommendation(
                        type="spot_instance",
                        severity="medium",
                        title="Use spot instances for fault-tolerant workloads",
                        description="Checkpoint-based training can use spot instances",
                        potential_savings_daily=spot_savings,
                        affected_resources=[team],
                    )
                )

            # Right-sizing recommendation
            if summary.avg_utilization < 50 and summary.gpu_count > 1:
                recommendations.append(
                    Recommendation(
                        type="right_sizing",
                        severity="medium",
                        title="Consider smaller GPU instances",
                        description=f"Average utilization is {summary.avg_utilization:.0f}%",
                        potential_savings_daily=summary.gpu_cost_daily * 0.3,
                        affected_resources=[team],
                    )
                )

            summary.recommendations = recommendations

    def get_recommendations(self) -> list[dict]:
        """Get all recommendations across teams."""
        if not self._cached_summary:
            self.calculate_costs()

        all_recommendations = []
        for team, summary in self._cached_summary.items():
            for rec in summary.recommendations:
                all_recommendations.append(
                    {
                        "team": team,
                        "type": rec.type,
                        "severity": rec.severity,
                        "title": rec.title,
                        "description": rec.description,
                        "potential_savings_daily": round(rec.potential_savings_daily, 2),
                        "affected_resources": rec.affected_resources,
                    }
                )

        # Sort by potential savings
        all_recommendations.sort(
            key=lambda x: x["potential_savings_daily"], reverse=True
        )
        return all_recommendations

    def get_cost_summary(self) -> dict:
        """Get cost summary for all teams."""
        if not self._cached_summary:
            self.calculate_costs()

        teams = {}
        total_gpu_cost = 0.0
        total_k8s_cost = 0.0
        total_idle_hours = 0.0

        for team, summary in self._cached_summary.items():
            teams[team] = {
                "gpu_cost_daily": round(summary.gpu_cost_daily, 2),
                "k8s_cost_daily": round(summary.k8s_cost_daily, 2),
                "total_cost_daily": round(summary.total_cost_daily, 2),
                "gpu_count": summary.gpu_count,
                "avg_utilization": round(summary.avg_utilization, 1),
                "idle_gpu_hours": round(summary.idle_gpu_hours, 1),
            }
            total_gpu_cost += summary.gpu_cost_daily
            total_k8s_cost += summary.k8s_cost_daily
            total_idle_hours += summary.idle_gpu_hours

        return {
            "timestamp": self._last_update.isoformat() if self._last_update else None,
            "totals": {
                "gpu_cost_daily": round(total_gpu_cost, 2),
                "k8s_cost_daily": round(total_k8s_cost, 2),
                "total_cost_daily": round(total_gpu_cost + total_k8s_cost, 2),
                "idle_gpu_hours": round(total_idle_hours, 1),
            },
            "teams": teams,
        }


# =============================================================================
# Global Instance
# =============================================================================

enricher: Optional[GPUEnricher] = None


def init_enricher():
    """Initialize the GPU enricher."""
    global enricher
    config_path = os.getenv("CONFIG_PATH", "/etc/config/config.yaml")
    prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    opencost_url = os.getenv("OPENCOST_URL", "http://opencost:9003")
    enricher = GPUEnricher(config_path, prometheus_url, opencost_url)


# =============================================================================
# API Endpoints
# =============================================================================


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/ready")
def ready():
    """Readiness check endpoint."""
    if enricher is None:
        return jsonify({"status": "not ready", "reason": "enricher not initialized"}), 503
    return jsonify({"status": "ready"})


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    if enricher:
        enricher.calculate_costs()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/api/v1/costs/summary")
def cost_summary():
    """Get cost summary by team.

    Returns:
        JSON with cost breakdown by team including GPU and Kubernetes costs.
    """
    if enricher is None:
        return jsonify({"error": "Enricher not initialized"}), 503

    try:
        summary = enricher.get_cost_summary()
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/costs/team/<team>")
def team_costs(team: str):
    """Get detailed costs for a specific team.

    Args:
        team: Team name

    Returns:
        JSON with detailed cost breakdown for the team.
    """
    if enricher is None:
        return jsonify({"error": "Enricher not initialized"}), 503

    try:
        summary = enricher.get_cost_summary()
        if team not in summary.get("teams", {}):
            return jsonify({"error": f"Team '{team}' not found"}), 404

        team_data = summary["teams"][team]
        team_data["team"] = team
        return jsonify(team_data)
    except Exception as e:
        logger.error(f"Error getting team costs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/recommendations")
def recommendations():
    """Get cost optimization recommendations.

    Query params:
        team: Filter by team (optional)
        severity: Filter by severity (optional)

    Returns:
        JSON with list of recommendations sorted by potential savings.
    """
    if enricher is None:
        return jsonify({"error": "Enricher not initialized"}), 503

    try:
        recs = enricher.get_recommendations()

        # Apply filters
        team_filter = request.args.get("team")
        severity_filter = request.args.get("severity")

        if team_filter:
            recs = [r for r in recs if r["team"] == team_filter]
        if severity_filter:
            recs = [r for r in recs if r["severity"] == severity_filter]

        total_savings = sum(r["potential_savings_daily"] for r in recs)

        return jsonify(
            {
                "recommendations": recs,
                "total_potential_savings_daily": round(total_savings, 2),
                "count": len(recs),
            }
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/gpu/utilization")
def gpu_utilization():
    """Get current GPU utilization metrics.

    Returns:
        JSON with GPU utilization by node and team.
    """
    if enricher is None:
        return jsonify({"error": "Enricher not initialized"}), 503

    try:
        gpu_metrics = enricher.get_gpu_metrics()
        return jsonify(
            {
                "gpus": [
                    {
                        "node": gpu.node,
                        "gpu_id": gpu.gpu_id,
                        "utilization": round(gpu.utilization, 1),
                        "memory_util": round(gpu.memory_util, 1),
                        "temperature": round(gpu.temperature, 1),
                        "team": gpu.team,
                        "model": gpu.model,
                        "is_idle": gpu.utilization < GPUEnricher.IDLE_THRESHOLD,
                    }
                    for gpu in gpu_metrics
                ],
                "count": len(gpu_metrics),
            }
        )
    except Exception as e:
        logger.error(f"Error getting GPU utilization: {e}")
        return jsonify({"error": str(e)}), 500


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    init_enricher()
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting AI FinOps GPU Enricher on port {port}")
    app.run(host="0.0.0.0", port=port)
