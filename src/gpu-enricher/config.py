"""
Centralized Configuration Module for AI FinOps Platform

Consolidates all configurable thresholds and settings in one place.
Values can be overridden via environment variables.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdConfig:
    """Configurable threshold values for the platform."""

    # GPU Utilization Thresholds
    idle_gpu_threshold: float = 20.0  # % - GPU considered idle below this
    underutilized_threshold: float = 30.0  # % - GPU underutilized below this
    overutilized_threshold: float = 85.0  # % - GPU overutilized above this
    memory_underutilized_threshold: float = 40.0  # % - Memory underutilized

    # Temperature Thresholds (Celsius)
    temp_warning_threshold: float = 75.0  # Warning if temp > this at low util
    temp_high_threshold: float = 85.0  # Alert threshold
    temp_critical_threshold: float = 90.0  # Critical threshold

    # Anomaly Detection Thresholds
    zscore_threshold: float = 2.5  # Standard deviations for anomaly
    iqr_multiplier: float = 1.5  # IQR multiplier for outliers
    cost_spike_threshold: float = 50.0  # % increase to flag as spike
    cost_drop_threshold: float = 30.0  # % decrease to flag as drop

    # Right-sizing Thresholds
    spot_eligible_threshold: float = 70.0  # % util for spot eligibility
    min_data_points: int = 24  # Minimum hours of data for recommendation
    high_confidence_threshold: float = 0.8

    # Rate Limiting
    default_rate_limit: int = 100  # Requests per minute
    notification_rate_limit: int = 30  # Max notifications per minute

    # Cost Calculation Settings
    min_cost_factor: float = 0.0  # Minimum cost factor for idle GPUs (0.0-1.0)
    # Set to 0.0 for accurate idle cost (shows $0 for truly idle)
    # Set to 0.1 for baseline cost assumption (10% minimum charge)

    # Budget Defaults
    default_team_budget: float = 5000.0  # Default monthly budget per team


# Default GPU Pricing Configuration
DEFAULT_GPU_PRICING = {
    "aws": {
        "g4dn.xlarge": {"on_demand": 0.526, "spot_avg": 0.158},
        "g4dn.2xlarge": {"on_demand": 0.752, "spot_avg": 0.226},
        "g4dn.4xlarge": {"on_demand": 1.204, "spot_avg": 0.361},
        "g5.xlarge": {"on_demand": 1.006, "spot_avg": 0.302},
        "g5.2xlarge": {"on_demand": 1.212, "spot_avg": 0.364},
        "p3.2xlarge": {"on_demand": 3.06, "spot_avg": 0.918},
        "p3.8xlarge": {"on_demand": 12.24, "spot_avg": 3.672},
        "p4d.24xlarge": {"on_demand": 32.77, "spot_avg": 9.831},
    },
    "gcp": {
        "n1-standard-4-t4": {"on_demand": 0.35, "spot_avg": 0.11},
        "a2-highgpu-1g": {"on_demand": 3.67, "spot_avg": 1.10},
    },
    "azure": {
        "Standard_NC4as_T4_v3": {"on_demand": 0.526, "spot_avg": 0.158},
        "Standard_NC6s_v3": {"on_demand": 3.06, "spot_avg": 0.918},
    },
}


def load_thresholds() -> ThresholdConfig:
    """Load thresholds from environment variables with defaults."""
    return ThresholdConfig(
        # GPU Utilization
        idle_gpu_threshold=float(os.getenv("THRESHOLD_IDLE_GPU", "20.0")),
        underutilized_threshold=float(os.getenv("THRESHOLD_UNDERUTILIZED", "30.0")),
        overutilized_threshold=float(os.getenv("THRESHOLD_OVERUTILIZED", "85.0")),
        memory_underutilized_threshold=float(
            os.getenv("THRESHOLD_MEMORY_UNDERUTILIZED", "40.0")
        ),
        # Temperature
        temp_warning_threshold=float(os.getenv("THRESHOLD_TEMP_WARNING", "75.0")),
        temp_high_threshold=float(os.getenv("THRESHOLD_TEMP_HIGH", "85.0")),
        temp_critical_threshold=float(os.getenv("THRESHOLD_TEMP_CRITICAL", "90.0")),
        # Anomaly Detection
        zscore_threshold=float(os.getenv("THRESHOLD_ZSCORE", "2.5")),
        iqr_multiplier=float(os.getenv("THRESHOLD_IQR_MULTIPLIER", "1.5")),
        cost_spike_threshold=float(os.getenv("THRESHOLD_COST_SPIKE", "50.0")),
        cost_drop_threshold=float(os.getenv("THRESHOLD_COST_DROP", "30.0")),
        # Right-sizing
        spot_eligible_threshold=float(os.getenv("THRESHOLD_SPOT_ELIGIBLE", "70.0")),
        min_data_points=int(os.getenv("THRESHOLD_MIN_DATA_POINTS", "24")),
        high_confidence_threshold=float(os.getenv("THRESHOLD_HIGH_CONFIDENCE", "0.8")),
        # Rate Limiting
        default_rate_limit=int(os.getenv("DEFAULT_RATE_LIMIT", "100")),
        notification_rate_limit=int(os.getenv("NOTIFICATION_RATE_LIMIT", "30")),
        # Cost Calculation
        min_cost_factor=float(os.getenv("MIN_COST_FACTOR", "0.0")),
        # Budget Defaults
        default_team_budget=float(os.getenv("DEFAULT_TEAM_BUDGET", "5000.0")),
    )


# Global configuration instance
thresholds = load_thresholds()


# Valid values for API parameters
VALID_SEVERITIES = {"low", "medium", "high", "info", "warning", "critical"}
VALID_PERIODS = {"1d", "7d", "30d", "90d"}
VALID_REPORT_FORMATS = {"csv", "pdf", "json"}
MAX_TEAM_NAME_LENGTH = 64
MAX_DAYS_QUERY = 365


class ValidationError(Exception):
    """Custom exception for input validation errors."""

    def __init__(self, message: str, field: str):
        self.message = message
        self.field = field
        super().__init__(message)


def validate_team_name(team: str) -> str:
    """Validate and sanitize team name."""
    if not team:
        raise ValidationError("Team name is required", "team")
    if len(team) > MAX_TEAM_NAME_LENGTH:
        raise ValidationError(
            f"Team name too long (max {MAX_TEAM_NAME_LENGTH} chars)", "team"
        )
    # Only allow alphanumeric, hyphens, and underscores
    sanitized = "".join(c for c in team if c.isalnum() or c in "-_")
    if not sanitized:
        raise ValidationError("Team name contains invalid characters", "team")
    return sanitized.lower()


def validate_severity(severity: str) -> str:
    """Validate severity parameter."""
    if severity and severity.lower() not in VALID_SEVERITIES:
        raise ValidationError(
            f"Invalid severity. Must be one of: {', '.join(VALID_SEVERITIES)}",
            "severity",
        )
    return severity.lower() if severity else severity


def validate_period(period: str) -> str:
    """Validate time period parameter."""
    if period not in VALID_PERIODS:
        raise ValidationError(
            f"Invalid period. Must be one of: {', '.join(VALID_PERIODS)}",
            "period",
        )
    return period


def validate_days(days: int) -> int:
    """Validate days parameter."""
    if days < 1:
        raise ValidationError("Days must be at least 1", "days")
    if days > MAX_DAYS_QUERY:
        raise ValidationError(f"Days cannot exceed {MAX_DAYS_QUERY}", "days")
    return days


def validate_min_savings(min_savings: float) -> float:
    """Validate minimum savings parameter."""
    if min_savings < 0:
        raise ValidationError("Minimum savings cannot be negative", "min_savings")
    return min_savings


def validate_report_format(format: str) -> str:
    """Validate report format parameter."""
    if format.lower() not in VALID_REPORT_FORMATS:
        raise ValidationError(
            f"Invalid format. Must be one of: {', '.join(VALID_REPORT_FORMATS)}",
            "format",
        )
    return format.lower()
