"""
Cloud Billing Integration Module for AI FinOps Platform

Integrates with cloud provider billing APIs to fetch actual cost data:
- AWS Cost Explorer API
- GCP Cloud Billing API
- Azure Cost Management API

This provides ground truth billing data to compare against estimated costs.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

# AWS Cost Explorer SDK
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

# GCP Cloud Billing SDK
from google.cloud import billing_v1
from google.api_core import exceptions as gcp_exceptions

# Azure Cost Management SDK
from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
from azure.core.exceptions import AzureError

logger = logging.getLogger(__name__)


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
                                        "Values": [
                                            "Amazon Elastic Compute Cloud - Compute"
                                        ],
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
                "utilization_percentage": float(total.get("UtilizationPercentage", 0)),
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


class GCPBilling:
    """GCP Cloud Billing API integration."""

    # GPU-related machine types
    GPU_MACHINE_PREFIXES = ["a2-", "g2-", "n1-"]
    GPU_ACCELERATOR_TYPES = ["nvidia-tesla-t4", "nvidia-tesla-a100", "nvidia-l4"]

    def __init__(
        self,
        project_id: str,
        billing_account_id: Optional[str] = None,
    ):
        self.project_id = project_id
        self.billing_account_id = billing_account_id
        self._client = None

    @property
    def client(self):
        """Lazy-load Cloud Billing client."""
        if self._client is None:
            self._client = billing_v1.CloudBillingClient()
        return self._client

    def get_gpu_costs_by_team(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        label_key: str = "ai-finops-team",
    ) -> dict[str, TeamBillingData]:
        """
        Get GPU-related costs grouped by team label.

        Note: This uses BigQuery export of billing data.
        Requires billing data export to be configured.

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)
            label_key: Label key for team identification

        Returns:
            Dictionary mapping team name to TeamBillingData
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Note: In production, this would query BigQuery billing export
        # GCP doesn't have a direct equivalent to AWS Cost Explorer
        # Billing data must be exported to BigQuery first
        logger.warning(
            "GCP billing requires BigQuery export configuration. "
            "See: https://cloud.google.com/billing/docs/how-to/export-data-bigquery"
        )

        # Return empty for now - implementation requires BigQuery setup
        return {}

    def get_sku_pricing(self, sku_id: str) -> Optional[dict]:
        """
        Get pricing information for a specific SKU.

        Args:
            sku_id: The SKU ID to look up

        Returns:
            Pricing information for the SKU
        """
        try:
            # List services to find Compute Engine
            services = self.client.list_services()
            compute_service = None
            for service in services:
                if "Compute Engine" in service.display_name:
                    compute_service = service
                    break

            if not compute_service:
                logger.error("Could not find Compute Engine service")
                return None

            # Get SKU details
            skus = self.client.list_skus(parent=compute_service.name)
            for sku in skus:
                if sku.sku_id == sku_id:
                    return {
                        "sku_id": sku.sku_id,
                        "description": sku.description,
                        "category": sku.category.resource_family,
                        "regions": list(sku.service_regions),
                    }

            return None

        except gcp_exceptions.GoogleAPIError as e:
            logger.error(f"GCP API error: {e}")
            return None


class AzureCostManagement:
    """Azure Cost Management API integration."""

    # GPU-related VM sizes
    GPU_VM_SIZES = ["Standard_NC", "Standard_ND", "Standard_NV", "Standard_NC_ads"]

    def __init__(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None,
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self._client = None
        self._credential = None

    @property
    def credential(self):
        """Lazy-load Azure credential."""
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        return self._credential

    @property
    def client(self):
        """Lazy-load Cost Management client."""
        if self._client is None:
            self._client = CostManagementClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
            )
        return self._client

    def get_gpu_costs_by_team(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tag_name: str = "ai-finops-team",
    ) -> dict[str, TeamBillingData]:
        """
        Get GPU-related costs grouped by team tag.

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)
            tag_name: Tag name for team identification

        Returns:
            Dictionary mapping team name to TeamBillingData
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        scope = f"/subscriptions/{self.subscription_id}"
        if self.resource_group:
            scope = f"{scope}/resourceGroups/{self.resource_group}"

        try:
            # Build query for GPU VMs grouped by team tag
            query_body = {
                "type": "ActualCost",
                "timeframe": "Custom",
                "timePeriod": {
                    "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                    "to": end_date.strftime("%Y-%m-%dT23:59:59Z"),
                },
                "dataset": {
                    "granularity": "Daily",
                    "aggregation": {
                        "totalCost": {"name": "Cost", "function": "Sum"},
                    },
                    "grouping": [
                        {"type": "TagKey", "name": tag_name},
                        {"type": "Dimension", "name": "MeterCategory"},
                    ],
                    "filter": {
                        "dimensions": {
                            "name": "MeterCategory",
                            "operator": "In",
                            "values": ["Virtual Machines"],
                        }
                    },
                },
            }

            result = self.client.query.usage(scope=scope, parameters=query_body)

            team_data: dict[str, TeamBillingData] = {}

            for row in result.rows:
                # Parse row data based on columns
                cost = float(row[0]) if row[0] else 0
                team = row[1] if len(row) > 1 and row[1] else "untagged"
                meter_category = row[2] if len(row) > 2 else ""

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
                if "GPU" in meter_category or any(
                    gpu in meter_category for gpu in self.GPU_VM_SIZES
                ):
                    team_data[team].gpu_cost += cost
                else:
                    team_data[team].compute_cost += cost

            return team_data

        except AzureError as e:
            logger.error(f"Azure Cost Management API error: {e}")
            return {}

    def get_reservation_utilization(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> dict:
        """
        Get Azure Reservation utilization data.

        Returns:
            Reservation utilization metrics
        """
        scope = f"/subscriptions/{self.subscription_id}"

        try:
            # Query reservation utilization
            # Note: Requires specific permissions and reservation purchases
            result = self.client.query.usage(
                scope=scope,
                parameters={
                    "type": "AmortizedCost",
                    "timeframe": "Custom",
                    "timePeriod": {
                        "from": start_date.strftime("%Y-%m-%dT00:00:00Z"),
                        "to": end_date.strftime("%Y-%m-%dT23:59:59Z"),
                    },
                    "dataset": {
                        "granularity": "Daily",
                        "aggregation": {
                            "totalCost": {"name": "Cost", "function": "Sum"},
                        },
                        "grouping": [
                            {"type": "Dimension", "name": "PricingModel"},
                        ],
                    },
                },
            )

            reservation_cost = 0
            on_demand_cost = 0

            for row in result.rows:
                cost = float(row[0]) if row[0] else 0
                pricing_model = row[1] if len(row) > 1 else ""

                if "Reservation" in pricing_model:
                    reservation_cost += cost
                else:
                    on_demand_cost += cost

            total_cost = reservation_cost + on_demand_cost
            utilization_pct = (
                (reservation_cost / total_cost * 100) if total_cost > 0 else 0
            )

            return {
                "utilization_percentage": round(utilization_pct, 1),
                "reservation_cost": round(reservation_cost, 2),
                "on_demand_cost": round(on_demand_cost, 2),
                "total_cost": round(total_cost, 2),
            }

        except AzureError as e:
            logger.error(f"Failed to fetch Azure reservation utilization: {e}")
            return {}


class BillingIntegration:
    """
    Unified billing integration interface.

    Abstracts cloud-specific implementations for AWS, GCP, and Azure.
    """

    def __init__(self):
        self._aws_explorer: Optional[AWSCostExplorer] = None
        self._gcp_billing: Optional[GCPBilling] = None
        self._azure_cost: Optional[AzureCostManagement] = None
        self._enabled = False
        self._enabled_providers: list[str] = []

    def initialize_aws(
        self,
        region: str = "us-east-1",
        profile_name: Optional[str] = None,
    ) -> bool:
        """Initialize AWS Cost Explorer integration."""
        try:
            self._aws_explorer = AWSCostExplorer(
                region=region,
                profile_name=profile_name,
            )
            # Test connection
            self._aws_explorer.client
            self._enabled = True
            self._enabled_providers.append("aws")
            logger.info("AWS Cost Explorer integration initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AWS integration: {e}")
            return False

    def initialize_gcp(
        self,
        project_id: str,
        billing_account_id: Optional[str] = None,
    ) -> bool:
        """Initialize GCP Cloud Billing integration."""
        try:
            self._gcp_billing = GCPBilling(
                project_id=project_id,
                billing_account_id=billing_account_id,
            )
            self._enabled = True
            self._enabled_providers.append("gcp")
            logger.info("GCP Cloud Billing integration initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GCP integration: {e}")
            return False

    def initialize_azure(
        self,
        subscription_id: str,
        resource_group: Optional[str] = None,
    ) -> bool:
        """Initialize Azure Cost Management integration."""
        try:
            self._azure_cost = AzureCostManagement(
                subscription_id=subscription_id,
                resource_group=resource_group,
            )
            # Test connection
            self._azure_cost.credential
            self._enabled = True
            self._enabled_providers.append("azure")
            logger.info("Azure Cost Management integration initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Azure integration: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        """Check if billing integration is enabled."""
        return self._enabled

    @property
    def enabled_providers(self) -> list[str]:
        """Get list of enabled cloud providers."""
        return self._enabled_providers.copy()

    def get_actual_costs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        provider: Optional[str] = None,
    ) -> dict[str, TeamBillingData]:
        """
        Get actual billing costs from cloud provider(s).

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: today)
            provider: Specific provider to query (aws, gcp, azure), or None for all

        Returns:
            Dictionary mapping team name to billing data
        """
        if not self._enabled:
            logger.warning("Billing integration not enabled")
            return {}

        all_data: dict[str, TeamBillingData] = {}

        # Query AWS
        if self._aws_explorer and (provider is None or provider == "aws"):
            aws_data = self._aws_explorer.get_gpu_costs_by_team(start_date, end_date)
            for team, data in aws_data.items():
                if team in all_data:
                    # Merge costs from multiple providers
                    all_data[team].total_cost += data.total_cost
                    all_data[team].gpu_cost += data.gpu_cost
                else:
                    all_data[team] = data

        # Query GCP
        if self._gcp_billing and (provider is None or provider == "gcp"):
            gcp_data = self._gcp_billing.get_gpu_costs_by_team(start_date, end_date)
            for team, data in gcp_data.items():
                if team in all_data:
                    all_data[team].total_cost += data.total_cost
                    all_data[team].gpu_cost += data.gpu_cost
                else:
                    all_data[team] = data

        # Query Azure
        if self._azure_cost and (provider is None or provider == "azure"):
            azure_data = self._azure_cost.get_gpu_costs_by_team(start_date, end_date)
            for team, data in azure_data.items():
                if team in all_data:
                    all_data[team].total_cost += data.total_cost
                    all_data[team].gpu_cost += data.gpu_cost
                else:
                    all_data[team] = data

        return all_data

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

            comparison.append(
                {
                    "team": team,
                    "estimated_daily": round(estimated, 2),
                    "actual_daily": round(actual_daily, 2),
                    "variance": round(variance, 2),
                    "variance_pct": round(variance_pct, 1),
                    "accuracy": round(100 - abs(variance_pct), 1)
                    if actual_daily
                    else None,
                }
            )

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
    # AWS Cost Explorer
    if os.getenv("ENABLE_AWS_BILLING", "false").lower() == "true":
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        aws_profile = os.getenv("AWS_PROFILE")
        billing_integration.initialize_aws(
            region=aws_region,
            profile_name=aws_profile,
        )

    # GCP Cloud Billing
    if os.getenv("ENABLE_GCP_BILLING", "false").lower() == "true":
        gcp_project = os.getenv("GCP_PROJECT_ID")
        gcp_billing_account = os.getenv("GCP_BILLING_ACCOUNT_ID")
        if gcp_project:
            billing_integration.initialize_gcp(
                project_id=gcp_project,
                billing_account_id=gcp_billing_account,
            )
        else:
            logger.warning("GCP_PROJECT_ID not set, skipping GCP billing integration")

    # Azure Cost Management
    if os.getenv("ENABLE_AZURE_BILLING", "false").lower() == "true":
        azure_subscription = os.getenv("AZURE_SUBSCRIPTION_ID")
        azure_resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        if azure_subscription:
            billing_integration.initialize_azure(
                subscription_id=azure_subscription,
                resource_group=azure_resource_group,
            )
        else:
            logger.warning(
                "AZURE_SUBSCRIPTION_ID not set, skipping Azure billing integration"
            )

    if billing_integration.is_enabled:
        logger.info(
            f"Billing integration enabled for: {billing_integration.enabled_providers}"
        )
