"""
Anomaly Detection Module for AI FinOps Platform

Implements statistical and ML-based anomaly detection for:
- Cost anomalies (unexpected spend spikes/drops)
- Utilization anomalies (unusual GPU usage patterns)
- Efficiency anomalies (degraded performance patterns)

Methods:
- Z-Score: Detects outliers based on standard deviations
- IQR (Interquartile Range): Robust outlier detection
- Moving Average: Trend-based anomaly detection
- Isolation Forest: ML-based anomaly detection (optional)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""

    COST_SPIKE = "cost_spike"
    COST_DROP = "cost_drop"
    UTILIZATION_SPIKE = "utilization_spike"
    UTILIZATION_DROP = "utilization_drop"
    EFFICIENCY_DEGRADATION = "efficiency_degradation"
    IDLE_PATTERN = "idle_pattern"
    TEMPERATURE_ANOMALY = "temperature_anomaly"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""

    type: AnomalyType
    severity: AnomalySeverity
    metric_name: str
    current_value: float
    expected_value: float
    deviation_pct: float
    team: str
    resource: str
    timestamp: datetime
    description: str
    recommendation: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 2),
            "expected_value": round(self.expected_value, 2),
            "deviation_pct": round(self.deviation_pct, 1),
            "team": self.team,
            "resource": self.resource,
            "timestamp": self.timestamp.isoformat(),
            "description": self.description,
            "recommendation": self.recommendation,
        }


class AnomalyDetector:
    """Statistical anomaly detection for FinOps metrics."""

    # Default thresholds
    ZSCORE_THRESHOLD = 2.5  # Standard deviations
    IQR_MULTIPLIER = 1.5  # IQR multiplier for outliers
    COST_SPIKE_THRESHOLD = 50  # % increase to flag as spike
    COST_DROP_THRESHOLD = 30  # % decrease to flag as drop
    UTILIZATION_LOW_THRESHOLD = 20  # % utilization considered low
    UTILIZATION_HIGH_THRESHOLD = 95  # % utilization considered high
    TEMP_HIGH_THRESHOLD = 85  # Celsius

    def __init__(
        self,
        zscore_threshold: float = ZSCORE_THRESHOLD,
        iqr_multiplier: float = IQR_MULTIPLIER,
    ):
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self._historical_data: dict[str, list[float]] = {}

    def add_data_point(self, metric_key: str, value: float) -> None:
        """Add a data point to historical data for trend analysis."""
        if metric_key not in self._historical_data:
            self._historical_data[metric_key] = []
        self._historical_data[metric_key].append(value)
        # Keep last 168 data points (7 days at hourly resolution)
        if len(self._historical_data[metric_key]) > 168:
            self._historical_data[metric_key] = self._historical_data[metric_key][-168:]

    def get_historical_data(self, metric_key: str) -> list[float]:
        """Get historical data for a metric."""
        return self._historical_data.get(metric_key, [])

    def detect_zscore_anomaly(
        self,
        values: list[float],
        current_value: float,
    ) -> tuple[bool, float, float]:
        """
        Detect anomaly using Z-score method.

        Returns:
            (is_anomaly, z_score, expected_value)
        """
        if len(values) < 3:
            return False, 0.0, current_value

        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return False, 0.0, mean

        z_score = (current_value - mean) / std
        is_anomaly = abs(z_score) > self.zscore_threshold

        return is_anomaly, z_score, mean

    def detect_iqr_anomaly(
        self,
        values: list[float],
        current_value: float,
    ) -> tuple[bool, float, float]:
        """
        Detect anomaly using IQR (Interquartile Range) method.

        More robust to outliers than Z-score.

        Returns:
            (is_anomaly, deviation_from_median, median)
        """
        if len(values) < 4:
            return False, 0.0, current_value

        arr = np.array(values)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        median = np.median(arr)

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        is_anomaly = current_value < lower_bound or current_value > upper_bound
        deviation = current_value - median

        return is_anomaly, deviation, median

    def detect_moving_average_anomaly(
        self,
        values: list[float],
        current_value: float,
        window_size: int = 24,
        threshold_pct: float = 30.0,
    ) -> tuple[bool, float, float]:
        """
        Detect anomaly using moving average comparison.

        Returns:
            (is_anomaly, deviation_pct, moving_avg)
        """
        if len(values) < window_size:
            return False, 0.0, current_value

        arr = np.array(values[-window_size:])
        moving_avg = np.mean(arr)

        if moving_avg == 0:
            return False, 0.0, moving_avg

        deviation_pct = ((current_value - moving_avg) / moving_avg) * 100
        is_anomaly = abs(deviation_pct) > threshold_pct

        return is_anomaly, deviation_pct, moving_avg

    def detect_cost_anomaly(
        self,
        team: str,
        current_cost: float,
        historical_costs: list[float],
    ) -> Optional[Anomaly]:
        """Detect cost anomalies for a team."""
        if not historical_costs:
            return None

        # Use IQR for robustness
        is_anomaly, deviation, expected = self.detect_iqr_anomaly(
            historical_costs, current_cost
        )

        if not is_anomaly:
            return None

        # Determine if spike or drop
        deviation_pct = ((current_cost - expected) / expected * 100) if expected else 0

        if deviation_pct > self.COST_SPIKE_THRESHOLD:
            return Anomaly(
                type=AnomalyType.COST_SPIKE,
                severity=AnomalySeverity.CRITICAL
                if deviation_pct > 100
                else AnomalySeverity.WARNING,
                metric_name="daily_cost",
                current_value=current_cost,
                expected_value=expected,
                deviation_pct=deviation_pct,
                team=team,
                resource="team_budget",
                timestamp=datetime.now(timezone.utc),
                description=f"Cost spike detected: ${current_cost:.2f} vs expected ${expected:.2f} ({deviation_pct:.0f}% increase)",
                recommendation="Review recent deployments and workload changes. Check for runaway jobs or misconfigured autoscaling.",
            )
        elif deviation_pct < -self.COST_DROP_THRESHOLD:
            return Anomaly(
                type=AnomalyType.COST_DROP,
                severity=AnomalySeverity.INFO,
                metric_name="daily_cost",
                current_value=current_cost,
                expected_value=expected,
                deviation_pct=deviation_pct,
                team=team,
                resource="team_budget",
                timestamp=datetime.now(timezone.utc),
                description=f"Unexpected cost drop: ${current_cost:.2f} vs expected ${expected:.2f} ({abs(deviation_pct):.0f}% decrease)",
                recommendation="Verify workloads are running as expected. Check for failed jobs or service outages.",
            )

        return None

    def detect_utilization_anomaly(
        self,
        node: str,
        gpu_id: str,
        team: str,
        current_util: float,
        historical_utils: list[float],
    ) -> Optional[Anomaly]:
        """Detect GPU utilization anomalies."""
        # Check for sustained low utilization (idle pattern)
        if current_util < self.UTILIZATION_LOW_THRESHOLD:
            if len(historical_utils) >= 6:  # At least 6 data points
                recent = historical_utils[-6:]
                if all(u < self.UTILIZATION_LOW_THRESHOLD for u in recent):
                    return Anomaly(
                        type=AnomalyType.IDLE_PATTERN,
                        severity=AnomalySeverity.WARNING,
                        metric_name="gpu_utilization",
                        current_value=current_util,
                        expected_value=50.0,  # Expected reasonable utilization
                        deviation_pct=-((50 - current_util) / 50 * 100),
                        team=team,
                        resource=f"{node}/gpu{gpu_id}",
                        timestamp=datetime.now(timezone.utc),
                        description=f"GPU {gpu_id} on {node} has been idle (<20% utilization) for extended period",
                        recommendation="Consider scaling down or terminating idle GPU resources to reduce costs.",
                    )

        # Check for unusual utilization spikes
        if historical_utils:
            is_anomaly, z_score, expected = self.detect_zscore_anomaly(
                historical_utils, current_util
            )
            if is_anomaly and z_score > 0:
                deviation_pct = (
                    ((current_util - expected) / expected * 100) if expected else 0
                )
                return Anomaly(
                    type=AnomalyType.UTILIZATION_SPIKE,
                    severity=AnomalySeverity.INFO,
                    metric_name="gpu_utilization",
                    current_value=current_util,
                    expected_value=expected,
                    deviation_pct=deviation_pct,
                    team=team,
                    resource=f"{node}/gpu{gpu_id}",
                    timestamp=datetime.now(timezone.utc),
                    description=f"Unusual utilization spike: {current_util:.1f}% vs typical {expected:.1f}%",
                    recommendation="Monitor for potential runaway processes or unexpected workload increases.",
                )

        return None

    def detect_temperature_anomaly(
        self,
        node: str,
        gpu_id: str,
        team: str,
        current_temp: float,
        utilization: float,
    ) -> Optional[Anomaly]:
        """Detect GPU temperature anomalies."""
        # High temperature regardless of utilization
        if current_temp > self.TEMP_HIGH_THRESHOLD:
            severity = (
                AnomalySeverity.CRITICAL
                if current_temp > 90
                else AnomalySeverity.WARNING
            )
            return Anomaly(
                type=AnomalyType.TEMPERATURE_ANOMALY,
                severity=severity,
                metric_name="gpu_temperature",
                current_value=current_temp,
                expected_value=70.0,  # Expected normal operating temp
                deviation_pct=((current_temp - 70) / 70 * 100),
                team=team,
                resource=f"{node}/gpu{gpu_id}",
                timestamp=datetime.now(timezone.utc),
                description=f"GPU temperature critical: {current_temp:.1f}C (threshold: {self.TEMP_HIGH_THRESHOLD}C)",
                recommendation="Check cooling systems and GPU workload. Consider throttling or migrating workloads.",
            )

        # High temperature with low utilization (potential cooling issue)
        if current_temp > 75 and utilization < 30:
            return Anomaly(
                type=AnomalyType.TEMPERATURE_ANOMALY,
                severity=AnomalySeverity.WARNING,
                metric_name="gpu_temperature",
                current_value=current_temp,
                expected_value=50.0,  # Expected temp at low utilization
                deviation_pct=((current_temp - 50) / 50 * 100),
                team=team,
                resource=f"{node}/gpu{gpu_id}",
                timestamp=datetime.now(timezone.utc),
                description=f"High temperature ({current_temp:.1f}C) despite low utilization ({utilization:.1f}%)",
                recommendation="Investigate potential cooling system issues or environmental factors.",
            )

        return None

    def detect_efficiency_anomaly(
        self,
        team: str,
        current_efficiency: float,
        historical_efficiency: list[float],
    ) -> Optional[Anomaly]:
        """Detect efficiency degradation."""
        if len(historical_efficiency) < 7:
            return None

        # Check for sustained efficiency drop
        recent_avg = np.mean(historical_efficiency[-7:])
        older_avg = (
            np.mean(historical_efficiency[:-7])
            if len(historical_efficiency) > 7
            else recent_avg
        )

        if older_avg > 0:
            degradation_pct = ((older_avg - recent_avg) / older_avg) * 100

            if degradation_pct > 20:  # 20% efficiency drop
                return Anomaly(
                    type=AnomalyType.EFFICIENCY_DEGRADATION,
                    severity=AnomalySeverity.WARNING,
                    metric_name="gpu_efficiency",
                    current_value=current_efficiency,
                    expected_value=older_avg,
                    deviation_pct=-degradation_pct,
                    team=team,
                    resource="team_efficiency",
                    timestamp=datetime.now(timezone.utc),
                    description=f"Efficiency degradation: current {current_efficiency:.1f}% vs historical {older_avg:.1f}% ({degradation_pct:.0f}% drop)",
                    recommendation="Review workload scheduling and resource allocation. Consider right-sizing GPU instances.",
                )

        return None

    def run_all_detections(
        self,
        team: str,
        metrics: dict,
    ) -> list[Anomaly]:
        """
        Run all anomaly detection methods on provided metrics.

        Args:
            team: Team name
            metrics: Dictionary containing:
                - current_cost: Current daily cost
                - historical_costs: List of historical daily costs
                - gpus: List of GPU metrics dicts
                - efficiency: Current efficiency score
                - historical_efficiency: List of historical efficiency scores

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Cost anomaly detection
        if "current_cost" in metrics and "historical_costs" in metrics:
            cost_anomaly = self.detect_cost_anomaly(
                team,
                metrics["current_cost"],
                metrics["historical_costs"],
            )
            if cost_anomaly:
                anomalies.append(cost_anomaly)

        # GPU metrics anomaly detection
        for gpu in metrics.get("gpus", []):
            # Get historical data for this GPU
            util_key = f"{team}:{gpu['node']}:{gpu['gpu_id']}:util"
            historical_utils = self.get_historical_data(util_key)

            # Add current data point
            self.add_data_point(util_key, gpu["utilization"])

            # Utilization anomaly
            util_anomaly = self.detect_utilization_anomaly(
                gpu["node"],
                gpu["gpu_id"],
                team,
                gpu["utilization"],
                historical_utils,
            )
            if util_anomaly:
                anomalies.append(util_anomaly)

            # Temperature anomaly
            temp_anomaly = self.detect_temperature_anomaly(
                gpu["node"],
                gpu["gpu_id"],
                team,
                gpu.get("temperature", 0),
                gpu["utilization"],
            )
            if temp_anomaly:
                anomalies.append(temp_anomaly)

        # Efficiency anomaly detection
        if "efficiency" in metrics and "historical_efficiency" in metrics:
            eff_anomaly = self.detect_efficiency_anomaly(
                team,
                metrics["efficiency"],
                metrics["historical_efficiency"],
            )
            if eff_anomaly:
                anomalies.append(eff_anomaly)

        return anomalies


# Global detector instance
anomaly_detector = AnomalyDetector()
