"""
Cloud Billing Integration Module for AI FinOps Platform

Integrates with cloud provider billing APIs to fetch actual cost data:
- AWS Cost Explorer API
- GCP Cloud Billing API (future)
- Azure Cost Management API (future)

This provides ground truth billing data to compare against estimated costs.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# Optional boto3 import for AWS integration
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    logger.warning("boto3 not available - AWS Cost Explorer integration disabled")


@dataclass
class BillingRecord:
    """Individual billing record from cloud provider."""

    date: datetime
    service: str
    resource_id: str
    resource_tags: dict
    usage_type: str
    usage_quantity: float
    usage_unit: str
    cost: float
    currency: str
    cloud_provider: str


@dataclass
class TeamBillingData:
    """Aggregated billing data for a team."""

    team: str
    period_start: datetime
    period_end: datetime
    total_cost: float
    gpu_cost: float
    compute_cost: float
    storage_cost: float
    network_cost: float
    other_cost: float
    currency: str
    records: list[BillingRecord]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "team": self.team,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost": round(self.total_cost, 2),
            "breakdown": {
                "gpu": round(self.gpu_cost, 2),
                "compute": round(self.compute_cost, 2),
                "storage": round(self.storage_cost, 2),
                "network": round(self.network_cost, 2),
                "other": round(self.other_cost, 2),
            },
            "currency": self.currency,
            "record_count": len(self.records),
        }


class AWSCostExplorer:
    """AWS Cost Explorer API integration."""

    # GPU-related instance types and services
    GPU_INSTANCE_PREFIXES = ["g4dn", "g5", "p3", "p4d", "p5", "inf1", "trn1"]
    GPU_SERVICES = ["Amazon Elastic Compute Cloud - Compute"]

    def __init__(
        self,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
    ):
        if not AWS_AVAILABLE:
            raise RuntimeError("boto3 is required for AWS Cost Explorer integration")

        self.region = region
        self.profile_name = profile_name
        self._client = None

    @property
    def client(self):
        """Lazy-load Cost Explorer client."""
        if self._client is None:
            try:
                session_kwargs = {}
                if self.profile_name:
                    session_kwargs["profile_name"] = self.profile_name

                session = boto3.Session(**session_kwargs)
                self._client = session.client("ce", region_name=self.region)
            except NoCredentialsError:
                logger.error("AWS credentials not configured")
                raise
        return self._client

    def get_cost_and_usage(
        self,
        start_date: datetime,
        end_date: datetime,
        granularity: str = "DAILY",
        group_by_tag: str = "ai-finops.io/team",
    ) -> dict:
        """
        Fetch cost and usage data from AWS Cost Explorer.

        Args:
            start_date: Start of period
            end_date: End of period
            granularity: DAILY, MONTHLY, or HOURLY
            group_by_tag: Tag to group costs by

        Returns:
            Cost Explorer response data
        """
        try:
            response = self.client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity=granularity,
                Metrics=["UnblendedCost", "UsageQuantity"],
                GroupBy=[
                    {"Type": "TAG", "Key": group_by_tag},
                    {"Type": "DIMENSION", "Key": "SERVICE"},
                ],
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": self.GPU_SERVICES,
                    }
                },
            )
            return response
        except ClientError as e:
            logger.error(f"AWS Cost Explorer API error: {e}")
            raise

    def get_gpu_costs_by_team(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tag_key: str = "ai-finops.io/team",
    ) -> dict[str, TeamBillingData]:
        """
        Get GPU-related costs grouped by team tag.

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)
            tag_key: Tag key for team identification

        Returns:
            Dictionary mapping team name to TeamBillingData
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        try:
            response = self.client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Metrics=["UnblendedCost", "UsageQuantity"],
                GroupBy=[
                    {"Type": "TAG", "Key": tag_key},
                ],
                Filter={
                    "Or": [
                        # GPU instances
                        {
                            "Dimensions": {
                                "Key": "INSTANCE_TYPE_FAMILY",
                                "Values": self.GPU_INSTANCE_PREFIXES,
                            }
                        },
                        # EC2 compute (for GPU nodes)
                        {
                            "And": [
                                {
                                    "Dimensions": {
                                        "Key": "SERVICE",
                                        "Values": ["Amazon Elastic Compute Cloud - Compute"],
                                    }
                                },
                                {
                                    "Tags": {
                                        "Key": "nvidia.com/gpu.present",
                                        "Values": ["true"],
                                    }
                                },
                            ]
                        },
                    ]
                },
            )
        except ClientError as e:
            logger.error(f"Failed to fetch GPU costs: {e}")
            return {}

        # Process response
        team_data: dict[str, TeamBillingData] = {}

        for result in response.get("ResultsByTime", []):
            for group in result.get("Groups", []):
                # Extract team from tag
                tag_value = group["Keys"][0]
                team = tag_value.replace(f"{tag_key}$", "") if tag_value else "untagged"

                if not team or team == f"{tag_key}$":
                    team = "untagged"

                cost = float(group["Metrics"]["UnblendedCost"]["Amount"])

                if team not in team_data:
                    team_data[team] = TeamBillingData(
                        team=team,
                        period_start=start_date,
                        period_end=end_date,
                        total_cost=0,
                        gpu_cost=0,
                        compute_cost=0,
                        storage_cost=0,
                        network_cost=0,
                        other_cost=0,
                        currency="USD",
                        records=[],
                    )

                team_data[team].total_cost += cost
                team_data[team].gpu_cost += cost  # All queried costs are GPU-related

        return team_data

    def get_cost_by_resource(
        self,
        start_date: datetime,
        end_date: datetime,
        resource_ids: Optional[list[str]] = None,
    ) -> list[BillingRecord]:
        """
        Get cost breakdown by individual resource.

        Note: Requires Cost Allocation Tags to be activated.

        Args:
            start_date: Start date
            end_date: End date
            resource_ids: Optional list of resource IDs to filter

        Returns:
            List of BillingRecord objects
        """
        try:
            filter_criteria = {
                "Dimensions": {
                    "Key": "SERVICE",
                    "Values": self.GPU_SERVICES,
                }
            }

            if resource_ids:
                filter_criteria = {
                    "And": [
                        filter_criteria,
                        {
                            "Dimensions": {
                                "Key": "RESOURCE_ID",
                                "Values": resource_ids,
                            }
                        },
                    ]
                }

            response = self.client.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Metrics=["UnblendedCost", "UsageQuantity"],
                GroupBy=[
                    {"Type": "DIMENSION", "Key": "RESOURCE_ID"},
                    {"Type": "DIMENSION", "Key": "USAGE_TYPE"},
                ],
                Filter=filter_criteria,
            )
        except ClientError as e:
            logger.error(f"Failed to fetch resource costs: {e}")
            return []

        records = []
        for result in response.get("ResultsByTime", []):
            date_str = result["TimePeriod"]["Start"]
            record_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )

            for group in result.get("Groups", []):
                keys = group["Keys"]
                resource_id = keys[0] if len(keys) > 0 else "unknown"
                usage_type = keys[1] if len(keys) > 1 else "unknown"

                cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
                usage_qty = float(group["Metrics"]["UsageQuantity"]["Amount"])

                records.append(
                    BillingRecord(
                        date=record_date,
                        service="EC2",
                        resource_id=resource_id,
                        resource_tags={},
                        usage_type=usage_type,
                        usage_quantity=usage_qty,
                        usage_unit="Hours",
                        cost=cost,
                        currency="USD",
                        cloud_provider="aws",
                    )
                )

        return records

    def get_savings_plans_utilization(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """
        Get Savings Plans utilization data.

        Returns:
            Savings Plans utilization metrics
        """
        try:
            response = self.client.get_savings_plans_utilization(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
            )

            total = response.get("Total", {})
            utilization = total.get("Utilization", {})

            return {
                "utilization_percentage": float(
                    utilization.get("UtilizationPercentage", 0)
                ),
                "total_commitment": float(
                    utilization.get("TotalCommitment", {}).get("Amount", 0)
                ),
                "used_commitment": float(
                    utilization.get("UsedCommitment", {}).get("Amount", 0)
                ),
                "unused_commitment": float(
                    utilization.get("UnusedCommitment", {}).get("Amount", 0)
                ),
            }
        except ClientError as e:
            logger.error(f"Failed to fetch Savings Plans utilization: {e}")
            return {}

    def get_reserved_instance_utilization(
        self,
        start_date: datetime,
        end_date: datetime,
        service: str = "Amazon Elastic Compute Cloud - Compute",
    ) -> dict:
        """
        Get Reserved Instance utilization data.

        Returns:
            RI utilization metrics
        """
        try:
            response = self.client.get_reservation_utilization(
                TimePeriod={
                    "Start": start_date.strftime("%Y-%m-%d"),
                    "End": end_date.strftime("%Y-%m-%d"),
                },
                Granularity="DAILY",
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": [service],
                    }
                },
            )

            total = response.get("Total", {})

            return {
                "utilization_percentage": float(
                    total.get("UtilizationPercentage", 0)
                ),
                "purchased_hours": float(total.get("PurchasedHours", 0)),
                "total_actual_hours": float(total.get("TotalActualHours", 0)),
                "unused_hours": float(total.get("UnusedHours", 0)),
                "on_demand_cost_equivalent": float(
                    total.get("OnDemandCostOfRIHoursUsed", 0)
                ),
                "net_ri_savings": float(total.get("NetRISavings", 0)),
            }
        except ClientError as e:
            logger.error(f"Failed to fetch RI utilization: {e}")
            return {}


class BillingIntegration:
    """
    Unified billing integration interface.

    Abstracts cloud-specific implementations.
    """

    def __init__(self):
        self._aws_explorer: Optional[AWSCostExplorer] = None
        self._enabled = False

    def initialize_aws(
        self,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
    ) -> bool:
        """Initialize AWS Cost Explorer integration."""
        if not AWS_AVAILABLE:
            logger.warning("AWS integration not available - boto3 not installed")
            return False

        try:
            self._aws_explorer = AWSCostExplorer(
                region=region,
                profile_name=profile_name,
            )
            # Test connection
            self._aws_explorer.client
            self._enabled = True
            logger.info("AWS Cost Explorer integration initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AWS integration: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        """Check if billing integration is enabled."""
        return self._enabled

    def get_actual_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, TeamBillingData]:
        """
        Get actual billing costs from cloud provider.

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)

        Returns:
            Dictionary mapping team name to billing data
        """
        if not self._enabled:
            logger.warning("Billing integration not enabled")
            return {}

        if self._aws_explorer:
            return self._aws_explorer.get_gpu_costs_by_team(start_date, end_date)

        return {}

    def get_cost_comparison(
        self,
        estimated_costs: dict[str, float],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """
        Compare estimated costs with actual billing data.

        Args:
            estimated_costs: Dictionary of team -> estimated daily cost
            start_date: Start date for actual costs
            end_date: End date for actual costs

        Returns:
            Comparison data with variance analysis
        """
        actual_data = self.get_actual_costs(start_date, end_date)

        if not actual_data:
            return {
                "status": "no_data",
                "message": "Actual billing data not available",
            }

        comparison = []
        total_estimated = 0
        total_actual = 0

        for team, estimated in estimated_costs.items():
            actual_team_data = actual_data.get(team)
            actual = actual_team_data.total_cost if actual_team_data else 0

            # Calculate daily average for comparison
            if actual_team_data:
                days = (
                    actual_team_data.period_end - actual_team_data.period_start
                ).days
                actual_daily = actual / max(days, 1)
            else:
                actual_daily = 0

            variance = estimated - actual_daily
            variance_pct = (variance / actual_daily * 100) if actual_daily else 0

            comparison.append({
                "team": team,
                "estimated_daily": round(estimated, 2),
                "actual_daily": round(actual_daily, 2),
                "variance": round(variance, 2),
                "variance_pct": round(variance_pct, 1),
                "accuracy": round(100 - abs(variance_pct), 1) if actual_daily else None,
            })

            total_estimated += estimated
            total_actual += actual_daily

        return {
            "status": "success",
            "comparison": comparison,
            "totals": {
                "estimated_daily": round(total_estimated, 2),
                "actual_daily": round(total_actual, 2),
                "variance": round(total_estimated - total_actual, 2),
            },
        }


# Global billing integration instance
billing_integration = BillingIntegration()


def initialize_billing():
    """Initialize billing integration from environment."""
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    aws_profile = os.getenv("AWS_PROFILE")

    if os.getenv("ENABLE_AWS_BILLING", "false").lower() == "true":
        billing_integration.initialize_aws(
            region=aws_region,
            profile_name=aws_profile,
        )