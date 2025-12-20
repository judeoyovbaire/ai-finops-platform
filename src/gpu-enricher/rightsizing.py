"""
Right-Sizing Recommendations Module for AI FinOps Platform

Analyzes GPU utilization patterns and recommends optimal instance types.

Strategies:
- Downsize: Recommend smaller instances for consistently underutilized GPUs
- Upsize: Recommend larger instances for consistently maxed-out GPUs
- Consolidate: Recommend combining workloads on fewer, larger instances
- Spot Migration: Recommend spot instances for fault-tolerant workloads
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RightsizeAction(Enum):
    """Types of right-sizing actions."""

    DOWNSIZE = "downsize"
    UPSIZE = "upsize"
    CONSOLIDATE = "consolidate"
    SPOT_MIGRATE = "spot_migrate"
    TERMINATE = "terminate"
    NO_ACTION = "no_action"


@dataclass
class InstanceSpec:
    """GPU instance specification."""

    instance_type: str
    cloud: str
    gpu_count: int
    gpu_model: str
    gpu_memory_gb: int
    vcpus: int
    memory_gb: int
    on_demand_hourly: float
    spot_hourly: float


# GPU Instance Catalog
INSTANCE_CATALOG = {
    # AWS GPU Instances
    "aws:g4dn.xlarge": InstanceSpec(
        instance_type="g4dn.xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="T4",
        gpu_memory_gb=16,
        vcpus=4,
        memory_gb=16,
        on_demand_hourly=0.526,
        spot_hourly=0.158,
    ),
    "aws:g4dn.2xlarge": InstanceSpec(
        instance_type="g4dn.2xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="T4",
        gpu_memory_gb=16,
        vcpus=8,
        memory_gb=32,
        on_demand_hourly=0.752,
        spot_hourly=0.226,
    ),
    "aws:g4dn.4xlarge": InstanceSpec(
        instance_type="g4dn.4xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="T4",
        gpu_memory_gb=16,
        vcpus=16,
        memory_gb=64,
        on_demand_hourly=1.204,
        spot_hourly=0.361,
    ),
    "aws:g4dn.12xlarge": InstanceSpec(
        instance_type="g4dn.12xlarge",
        cloud="aws",
        gpu_count=4,
        gpu_model="T4",
        gpu_memory_gb=64,
        vcpus=48,
        memory_gb=192,
        on_demand_hourly=3.912,
        spot_hourly=1.174,
    ),
    "aws:g5.xlarge": InstanceSpec(
        instance_type="g5.xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="A10G",
        gpu_memory_gb=24,
        vcpus=4,
        memory_gb=16,
        on_demand_hourly=1.006,
        spot_hourly=0.302,
    ),
    "aws:g5.2xlarge": InstanceSpec(
        instance_type="g5.2xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="A10G",
        gpu_memory_gb=24,
        vcpus=8,
        memory_gb=32,
        on_demand_hourly=1.212,
        spot_hourly=0.364,
    ),
    "aws:g5.4xlarge": InstanceSpec(
        instance_type="g5.4xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="A10G",
        gpu_memory_gb=24,
        vcpus=16,
        memory_gb=64,
        on_demand_hourly=1.624,
        spot_hourly=0.487,
    ),
    "aws:g5.12xlarge": InstanceSpec(
        instance_type="g5.12xlarge",
        cloud="aws",
        gpu_count=4,
        gpu_model="A10G",
        gpu_memory_gb=96,
        vcpus=48,
        memory_gb=192,
        on_demand_hourly=5.672,
        spot_hourly=1.702,
    ),
    "aws:p3.2xlarge": InstanceSpec(
        instance_type="p3.2xlarge",
        cloud="aws",
        gpu_count=1,
        gpu_model="V100",
        gpu_memory_gb=16,
        vcpus=8,
        memory_gb=61,
        on_demand_hourly=3.06,
        spot_hourly=0.918,
    ),
    "aws:p3.8xlarge": InstanceSpec(
        instance_type="p3.8xlarge",
        cloud="aws",
        gpu_count=4,
        gpu_model="V100",
        gpu_memory_gb=64,
        vcpus=32,
        memory_gb=244,
        on_demand_hourly=12.24,
        spot_hourly=3.672,
    ),
    "aws:p4d.24xlarge": InstanceSpec(
        instance_type="p4d.24xlarge",
        cloud="aws",
        gpu_count=8,
        gpu_model="A100",
        gpu_memory_gb=320,
        vcpus=96,
        memory_gb=1152,
        on_demand_hourly=32.77,
        spot_hourly=9.831,
    ),
}


@dataclass
class RightsizeRecommendation:
    """Right-sizing recommendation."""

    action: RightsizeAction
    current_instance: str
    recommended_instance: Optional[str]
    team: str
    node: str
    reason: str
    current_cost_daily: float
    projected_cost_daily: float
    savings_daily: float
    savings_pct: float
    confidence: float  # 0-1 confidence score
    utilization_avg: float
    utilization_peak: float
    memory_util_avg: float
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "action": self.action.value,
            "current_instance": self.current_instance,
            "recommended_instance": self.recommended_instance,
            "team": self.team,
            "node": self.node,
            "reason": self.reason,
            "current_cost_daily": round(self.current_cost_daily, 2),
            "projected_cost_daily": round(self.projected_cost_daily, 2),
            "savings_daily": round(self.savings_daily, 2),
            "savings_pct": round(self.savings_pct, 1),
            "confidence": round(self.confidence, 2),
            "utilization_avg": round(self.utilization_avg, 1),
            "utilization_peak": round(self.utilization_peak, 1),
            "memory_util_avg": round(self.memory_util_avg, 1),
            "timestamp": self.timestamp.isoformat(),
        }


class RightsizingEngine:
    """Engine for generating right-sizing recommendations."""

    # Thresholds for recommendations
    UNDERUTILIZED_THRESHOLD = 30  # % average utilization
    OVERUTILIZED_THRESHOLD = 85  # % average utilization
    IDLE_THRESHOLD = 10  # % average utilization for termination
    MEMORY_UNDERUTILIZED_THRESHOLD = 40  # % memory utilization
    SPOT_ELIGIBLE_THRESHOLD = 70  # % utilization threshold for spot
    MIN_DATA_POINTS = 24  # Minimum hours of data for recommendation
    HIGH_CONFIDENCE_THRESHOLD = 0.8

    def __init__(self, catalog: dict = None):
        self.catalog = catalog or INSTANCE_CATALOG
        self._utilization_history: dict[str, list[float]] = {}
        self._memory_history: dict[str, list[float]] = {}

    def add_utilization_sample(
        self,
        node: str,
        gpu_id: str,
        utilization: float,
        memory_util: float,
    ) -> None:
        """Add utilization sample for trend analysis."""
        key = f"{node}:{gpu_id}"

        if key not in self._utilization_history:
            self._utilization_history[key] = []
            self._memory_history[key] = []

        self._utilization_history[key].append(utilization)
        self._memory_history[key].append(memory_util)

        # Keep last 168 samples (7 days hourly)
        if len(self._utilization_history[key]) > 168:
            self._utilization_history[key] = self._utilization_history[key][-168:]
            self._memory_history[key] = self._memory_history[key][-168:]

    def get_utilization_stats(
        self,
        node: str,
        gpu_id: str,
    ) -> tuple[float, float, float, float]:
        """
        Get utilization statistics.

        Returns:
            (avg_util, peak_util, avg_memory, data_points)
        """
        key = f"{node}:{gpu_id}"
        util_data = self._utilization_history.get(key, [])
        mem_data = self._memory_history.get(key, [])

        if not util_data:
            return 0, 0, 0, 0

        return (
            np.mean(util_data),
            np.max(util_data),
            np.mean(mem_data) if mem_data else 0,
            len(util_data),
        )

    def find_smaller_instance(self, current: str) -> Optional[str]:
        """Find a smaller instance type in the same family."""
        if current not in self.catalog:
            return None

        current_spec = self.catalog[current]

        # Find instances in same family with lower cost
        candidates = []
        for key, spec in self.catalog.items():
            if (
                spec.cloud == current_spec.cloud
                and spec.gpu_model == current_spec.gpu_model
                and spec.on_demand_hourly < current_spec.on_demand_hourly
                and spec.gpu_count <= current_spec.gpu_count
            ):
                candidates.append((key, spec))

        if not candidates:
            return None

        # Return the largest of the smaller options
        candidates.sort(key=lambda x: x[1].on_demand_hourly, reverse=True)
        return candidates[0][0]

    def find_larger_instance(self, current: str) -> Optional[str]:
        """Find a larger instance type in the same family."""
        if current not in self.catalog:
            return None

        current_spec = self.catalog[current]

        # Find instances in same family with higher capacity
        candidates = []
        for key, spec in self.catalog.items():
            if (
                spec.cloud == current_spec.cloud
                and spec.gpu_model == current_spec.gpu_model
                and spec.on_demand_hourly > current_spec.on_demand_hourly
            ):
                candidates.append((key, spec))

        if not candidates:
            return None

        # Return the smallest of the larger options
        candidates.sort(key=lambda x: x[1].on_demand_hourly)
        return candidates[0][0]

    def analyze_gpu(
        self,
        node: str,
        gpu_id: str,
        team: str,
        instance_type: str,
        is_spot_tolerant: bool = False,
    ) -> Optional[RightsizeRecommendation]:
        """
        Analyze a GPU and generate right-sizing recommendation.

        Args:
            node: Node name
            gpu_id: GPU ID
            team: Team name
            instance_type: Current instance type (e.g., "aws:g4dn.xlarge")
            is_spot_tolerant: Whether workload can tolerate spot interruptions

        Returns:
            RightsizeRecommendation or None if no action needed
        """
        avg_util, peak_util, avg_memory, data_points = self.get_utilization_stats(
            node, gpu_id
        )

        if data_points < self.MIN_DATA_POINTS:
            logger.debug(
                f"Insufficient data for {node}/gpu{gpu_id}: {data_points} points"
            )
            return None

        # Calculate confidence based on data points and consistency
        confidence = min(1.0, data_points / 168)  # Max confidence at 7 days

        # Get current instance spec
        instance_key = instance_type if ":" in instance_type else f"aws:{instance_type}"
        current_spec = self.catalog.get(instance_key)
        if not current_spec:
            logger.warning(f"Unknown instance type: {instance_type}")
            return None

        current_daily_cost = current_spec.on_demand_hourly * 24

        # Check for termination (idle)
        if avg_util < self.IDLE_THRESHOLD and peak_util < 20:
            return RightsizeRecommendation(
                action=RightsizeAction.TERMINATE,
                current_instance=instance_type,
                recommended_instance=None,
                team=team,
                node=node,
                reason=f"GPU consistently idle (avg: {avg_util:.1f}%, peak: {peak_util:.1f}%)",
                current_cost_daily=current_daily_cost,
                projected_cost_daily=0,
                savings_daily=current_daily_cost,
                savings_pct=100,
                confidence=confidence,
                utilization_avg=avg_util,
                utilization_peak=peak_util,
                memory_util_avg=avg_memory,
                timestamp=datetime.now(timezone.utc),
            )

        # Check for downsize
        if (
            avg_util < self.UNDERUTILIZED_THRESHOLD
            and avg_memory < self.MEMORY_UNDERUTILIZED_THRESHOLD
        ):
            smaller = self.find_smaller_instance(instance_key)
            if smaller:
                smaller_spec = self.catalog[smaller]
                projected_cost = smaller_spec.on_demand_hourly * 24
                savings = current_daily_cost - projected_cost
                return RightsizeRecommendation(
                    action=RightsizeAction.DOWNSIZE,
                    current_instance=instance_type,
                    recommended_instance=smaller_spec.instance_type,
                    team=team,
                    node=node,
                    reason=f"Underutilized GPU (avg: {avg_util:.1f}%, memory: {avg_memory:.1f}%)",
                    current_cost_daily=current_daily_cost,
                    projected_cost_daily=projected_cost,
                    savings_daily=savings,
                    savings_pct=(savings / current_daily_cost) * 100,
                    confidence=confidence,
                    utilization_avg=avg_util,
                    utilization_peak=peak_util,
                    memory_util_avg=avg_memory,
                    timestamp=datetime.now(timezone.utc),
                )

        # Check for upsize
        if avg_util > self.OVERUTILIZED_THRESHOLD:
            larger = self.find_larger_instance(instance_key)
            if larger:
                larger_spec = self.catalog[larger]
                projected_cost = larger_spec.on_demand_hourly * 24
                # Negative savings (cost increase) but worth it for performance
                cost_increase = projected_cost - current_daily_cost
                return RightsizeRecommendation(
                    action=RightsizeAction.UPSIZE,
                    current_instance=instance_type,
                    recommended_instance=larger_spec.instance_type,
                    team=team,
                    node=node,
                    reason=f"GPU overutilized (avg: {avg_util:.1f}%, peak: {peak_util:.1f}%)",
                    current_cost_daily=current_daily_cost,
                    projected_cost_daily=projected_cost,
                    savings_daily=-cost_increase,
                    savings_pct=-(cost_increase / current_daily_cost) * 100,
                    confidence=confidence,
                    utilization_avg=avg_util,
                    utilization_peak=peak_util,
                    memory_util_avg=avg_memory,
                    timestamp=datetime.now(timezone.utc),
                )

        # Check for spot migration
        if (
            is_spot_tolerant
            and avg_util < self.SPOT_ELIGIBLE_THRESHOLD
            and current_spec.spot_hourly < current_spec.on_demand_hourly
        ):
            spot_daily_cost = current_spec.spot_hourly * 24
            savings = current_daily_cost - spot_daily_cost
            return RightsizeRecommendation(
                action=RightsizeAction.SPOT_MIGRATE,
                current_instance=instance_type,
                recommended_instance=instance_type,  # Same instance, spot pricing
                team=team,
                node=node,
                reason=f"Workload suitable for spot instances (avg util: {avg_util:.1f}%)",
                current_cost_daily=current_daily_cost,
                projected_cost_daily=spot_daily_cost,
                savings_daily=savings,
                savings_pct=(savings / current_daily_cost) * 100,
                confidence=confidence * 0.9,  # Slightly lower confidence for spot
                utilization_avg=avg_util,
                utilization_peak=peak_util,
                memory_util_avg=avg_memory,
                timestamp=datetime.now(timezone.utc),
            )

        return None

    def analyze_team(
        self,
        team: str,
        gpus: list[dict],
    ) -> list[RightsizeRecommendation]:
        """
        Analyze all GPUs for a team and generate recommendations.

        Args:
            team: Team name
            gpus: List of GPU dicts with node, gpu_id, instance_type, utilization, memory_util

        Returns:
            List of right-sizing recommendations
        """
        recommendations = []

        for gpu in gpus:
            # Add sample to history
            self.add_utilization_sample(
                gpu["node"],
                gpu["gpu_id"],
                gpu["utilization"],
                gpu.get("memory_util", 0),
            )

            # Generate recommendation
            rec = self.analyze_gpu(
                node=gpu["node"],
                gpu_id=gpu["gpu_id"],
                team=team,
                instance_type=gpu.get("instance_type", "aws:g4dn.xlarge"),
                is_spot_tolerant=gpu.get("spot_tolerant", False),
            )

            if rec:
                recommendations.append(rec)

        # Sort by savings potential
        recommendations.sort(key=lambda x: x.savings_daily, reverse=True)

        return recommendations

    def get_summary(
        self,
        recommendations: list[RightsizeRecommendation],
    ) -> dict:
        """Generate summary of all recommendations."""
        if not recommendations:
            return {
                "total_recommendations": 0,
                "potential_savings_daily": 0,
                "potential_savings_monthly": 0,
                "by_action": {},
            }

        by_action = {}
        for rec in recommendations:
            action = rec.action.value
            if action not in by_action:
                by_action[action] = {"count": 0, "savings_daily": 0}
            by_action[action]["count"] += 1
            by_action[action]["savings_daily"] += rec.savings_daily

        total_savings = sum(r.savings_daily for r in recommendations)

        return {
            "total_recommendations": len(recommendations),
            "potential_savings_daily": round(total_savings, 2),
            "potential_savings_monthly": round(total_savings * 30, 2),
            "by_action": by_action,
        }


# Global engine instance
rightsizing_engine = RightsizingEngine()
