"""
AI FinOps Cost Calculator Service

Calculates GPU costs based on utilization metrics and cloud pricing.
Exposes Prometheus metrics for cost attribution.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import yaml
from flask import Flask, jsonify
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
import requests

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Prometheus metrics
gpu_cost_hourly = Gauge(
    "ai_finops_gpu_cost_per_hour",
    "Estimated hourly GPU cost in USD",
    ["node", "instance_type", "team"],
)

inference_cost_total = Counter(
    "ai_finops_inference_cost_total",
    "Total inference cost in USD",
    ["model", "team"],
)

team_cost_daily = Gauge(
    "ai_finops_team_cost_daily",
    "Daily cost by team in USD",
    ["team"],
)

idle_gpu_hours = Gauge(
    "ai_finops_idle_gpu_hours",
    "Hours of idle GPU time (utilization < 20%)",
    ["node", "team"],
)

spot_savings_potential = Gauge(
    "ai_finops_spot_savings_potential",
    "Potential savings from using spot instances",
    ["node", "instance_type"],
)


@dataclass
class GPUPricing:
    """GPU instance pricing configuration."""

    on_demand: float
    spot_avg: float


class CostCalculator:
    """Calculate GPU costs based on utilization and pricing."""

    def __init__(self, config_path: str, prometheus_url: str):
        self.prometheus_url = prometheus_url
        self.pricing = self._load_pricing(config_path)
        logger.info(f"Loaded pricing for {len(self.pricing)} instance types")

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
        except Exception as e:
            logger.error(f"Failed to load pricing config: {e}")
            return {}

    def query_prometheus(self, query: str) -> list[dict]:
        """Query Prometheus for metrics."""
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data", {}).get("result", [])
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return []

    def calculate_costs(self) -> None:
        """Calculate and update cost metrics."""
        # Get GPU utilization
        gpu_util = self.query_prometheus("DCGM_FI_DEV_GPU_UTIL")

        team_costs: dict[str, float] = {}
        team_idle_hours: dict[str, float] = {}

        for metric in gpu_util:
            labels = metric.get("metric", {})
            value = float(metric.get("value", [0, 0])[1])

            node = labels.get("node", "unknown")
            team = labels.get("team", "unknown")
            instance_type = labels.get("instance_type", "g4dn.xlarge")

            # Get pricing (default to AWS g4dn.xlarge)
            pricing_key = f"aws:{instance_type}"
            pricing = self.pricing.get(pricing_key)
            if not pricing:
                pricing = GPUPricing(on_demand=0.526, spot_avg=0.158)

            # Calculate hourly cost based on utilization
            # Full cost when utilized, reduced when idle
            utilization_factor = value / 100.0
            hourly_cost = pricing.on_demand * max(utilization_factor, 0.1)

            # Update metrics
            gpu_cost_hourly.labels(
                node=node, instance_type=instance_type, team=team
            ).set(hourly_cost)

            # Track team costs (daily estimate = hourly * 24)
            if team not in team_costs:
                team_costs[team] = 0
            team_costs[team] += hourly_cost * 24

            # Track idle GPU hours
            if value < 20:
                if team not in team_idle_hours:
                    team_idle_hours[team] = 0
                team_idle_hours[team] += 1
                idle_gpu_hours.labels(node=node, team=team).set(1)

            # Calculate spot savings potential
            potential_savings = (pricing.on_demand - pricing.spot_avg) * 24
            spot_savings_potential.labels(
                node=node, instance_type=instance_type
            ).set(potential_savings)

        # Update team daily costs
        for team, cost in team_costs.items():
            team_cost_daily.labels(team=team).set(cost)

        logger.info(f"Updated costs for {len(team_costs)} teams")


# Global calculator instance
calculator: Optional[CostCalculator] = None


def init_calculator():
    """Initialize the cost calculator."""
    global calculator
    config_path = os.getenv("CONFIG_PATH", "/etc/config/config.yaml")
    prometheus_url = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
    calculator = CostCalculator(config_path, prometheus_url)


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/ready")
def ready():
    """Readiness check endpoint."""
    if calculator is None:
        return jsonify({"status": "not ready"}), 503
    return jsonify({"status": "ready"})


@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    if calculator:
        calculator.calculate_costs()
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/api/v1/costs/summary")
def cost_summary():
    """Get cost summary by team."""
    if calculator is None:
        return jsonify({"error": "Calculator not initialized"}), 503

    # Query current costs
    calculator.calculate_costs()

    # Build summary from metrics
    summary = {
        "teams": {},
        "total_daily_cost": 0,
        "idle_gpu_hours": 0,
    }

    # This would be populated from actual metric values
    return jsonify(summary)


@app.route("/api/v1/recommendations")
def recommendations():
    """Get cost optimization recommendations."""
    recs = [
        {
            "type": "spot_instance",
            "description": "Switch to spot instances for non-critical workloads",
            "potential_savings": "60-70%",
            "affected_nodes": [],
        },
        {
            "type": "idle_gpu",
            "description": "Scale down idle GPU nodes",
            "potential_savings": "100% of idle cost",
            "affected_nodes": [],
        },
        {
            "type": "right_sizing",
            "description": "Use smaller GPU instances for low-memory workloads",
            "potential_savings": "30-50%",
            "affected_nodes": [],
        },
    ]
    return jsonify({"recommendations": recs})


if __name__ == "__main__":
    init_calculator()
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting AI FinOps Cost Calculator on port {port}")
    app.run(host="0.0.0.0", port=port)