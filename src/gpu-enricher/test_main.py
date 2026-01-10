"""
Unit tests for AI FinOps GPU Enricher Service
"""

import json
import os
import tempfile

import pytest
import responses
import yaml

from main import (
    app,
    GPUEnricher,
    GPUPricing,
    GPUMetrics,
    TeamCostSummary,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_config():
    """Create a sample pricing configuration."""
    return {
        "pricing": {
            "aws": {
                "g4dn.xlarge": {"on_demand": 0.526, "spot_avg": 0.158},
                "g4dn.2xlarge": {"on_demand": 0.752, "spot_avg": 0.226},
                "p3.2xlarge": {"on_demand": 3.06, "spot_avg": 0.918},
            },
            "gcp": {
                "n1-standard-4-t4": {"on_demand": 0.35, "spot_avg": 0.11},
            },
        },
        "thresholds": {
            "idle_gpu_utilization": 20,
        },
    }


@pytest.fixture
def config_file(sample_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config, f)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def enricher(config_file):
    """Create a GPUEnricher instance with test configuration."""
    return GPUEnricher(
        config_path=config_file,
        prometheus_url="http://localhost:9090",
        opencost_url="http://localhost:9003",
    )


@pytest.fixture
def mock_prometheus_gpu_util():
    """Mock Prometheus GPU utilization response."""
    return {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {
                    "metric": {
                        "node": "gpu-node-1",
                        "gpu": "0",
                        "team": "ml-platform",
                        "model": "recommendation",
                        "instance_type": "g4dn.xlarge",
                    },
                    "value": [1699999999, "75"],
                },
                {
                    "metric": {
                        "node": "gpu-node-1",
                        "gpu": "1",
                        "team": "ml-platform",
                        "model": "recommendation",
                        "instance_type": "g4dn.xlarge",
                    },
                    "value": [1699999999, "15"],  # Idle GPU
                },
                {
                    "metric": {
                        "node": "gpu-node-2",
                        "gpu": "0",
                        "team": "data-science",
                        "model": "nlp-classifier",
                        "instance_type": "p3.2xlarge",
                    },
                    "value": [1699999999, "90"],
                },
            ],
        },
    }


@pytest.fixture
def mock_opencost_response():
    """Mock OpenCost allocation response."""
    return {
        "code": 200,
        "data": [
            {
                "ml-platform/deployment": {
                    "name": "ml-platform/deployment",
                    "properties": {
                        "namespace": "ml-platform",
                        "controller": "deployment",
                    },
                    "totalCost": 25.50,
                    "cpuCost": 15.00,
                    "memoryCost": 10.50,
                },
                "data-science/deployment": {
                    "name": "data-science/deployment",
                    "properties": {
                        "namespace": "data-science",
                        "controller": "deployment",
                    },
                    "totalCost": 18.75,
                    "cpuCost": 10.00,
                    "memoryCost": 8.75,
                },
            }
        ],
    }


# =============================================================================
# GPUPricing Tests
# =============================================================================


class TestGPUPricing:
    def test_spot_savings_percentage(self):
        pricing = GPUPricing(on_demand=1.00, spot_avg=0.30)
        assert pricing.spot_savings_pct == 70.0

    def test_spot_savings_percentage_zero_on_demand(self):
        pricing = GPUPricing(on_demand=0.0, spot_avg=0.30)
        assert pricing.spot_savings_pct == 0.0

    def test_pricing_values(self):
        pricing = GPUPricing(on_demand=0.526, spot_avg=0.158)
        assert pricing.on_demand == 0.526
        assert pricing.spot_avg == 0.158


# =============================================================================
# GPUMetrics Tests
# =============================================================================


class TestGPUMetrics:
    def test_gpu_metrics_creation(self):
        metrics = GPUMetrics(
            node="gpu-node-1",
            gpu_id="0",
            utilization=75.5,
            memory_util=60.0,
            temperature=65.0,
            power_usage=200.0,
            team="ml-platform",
            model="recommendation",
            instance_type="g4dn.xlarge",
        )
        assert metrics.node == "gpu-node-1"
        assert metrics.utilization == 75.5
        assert metrics.team == "ml-platform"

    def test_gpu_metrics_defaults(self):
        metrics = GPUMetrics(
            node="node-1",
            gpu_id="0",
            utilization=50.0,
            memory_util=40.0,
            temperature=55.0,
            power_usage=150.0,
        )
        assert metrics.team == "unknown"
        assert metrics.model == "unknown"
        assert metrics.instance_type == "unknown"


# =============================================================================
# TeamCostSummary Tests
# =============================================================================


class TestTeamCostSummary:
    def test_total_cost_calculation(self):
        summary = TeamCostSummary(
            team="ml-platform",
            gpu_cost_daily=100.0,
            k8s_cost_daily=25.0,
        )
        assert summary.total_cost_daily == 125.0

    def test_default_values(self):
        summary = TeamCostSummary(team="test-team")
        assert summary.gpu_cost_daily == 0.0
        assert summary.k8s_cost_daily == 0.0
        assert summary.idle_gpu_hours == 0.0
        assert summary.gpu_count == 0
        assert summary.recommendations == []


# =============================================================================
# GPUEnricher Tests
# =============================================================================


class TestGPUEnricher:
    def test_load_pricing_from_config(self, enricher):
        assert "aws:g4dn.xlarge" in enricher.pricing
        assert enricher.pricing["aws:g4dn.xlarge"].on_demand == 0.526
        assert enricher.pricing["aws:g4dn.xlarge"].spot_avg == 0.158

    def test_load_pricing_missing_file(self):
        enricher = GPUEnricher(
            config_path="/nonexistent/config.yaml",
            prometheus_url="http://localhost:9090",
            opencost_url="http://localhost:9003",
        )
        # Should use default pricing
        assert "aws:g4dn.xlarge" in enricher.pricing

    def test_default_pricing(self, enricher):
        defaults = enricher._default_pricing()
        assert "aws:g4dn.xlarge" in defaults
        assert "aws:p3.2xlarge" in defaults

    @responses.activate
    def test_query_prometheus_success(self, enricher, mock_prometheus_gpu_util):
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json=mock_prometheus_gpu_util,
            status=200,
        )

        result = enricher.query_prometheus("DCGM_FI_DEV_GPU_UTIL")
        assert len(result) == 3
        assert result[0]["metric"]["node"] == "gpu-node-1"

    @responses.activate
    def test_query_prometheus_timeout(self, enricher):
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            body=Exception("Connection timeout"),
        )

        result = enricher.query_prometheus("DCGM_FI_DEV_GPU_UTIL")
        assert result == []

    @responses.activate
    def test_query_prometheus_error(self, enricher):
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            status=500,
        )

        result = enricher.query_prometheus("DCGM_FI_DEV_GPU_UTIL")
        assert result == []

    @responses.activate
    def test_query_opencost_success(self, enricher, mock_opencost_response):
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            json=mock_opencost_response,
            status=200,
        )

        result = enricher.query_opencost("1d")
        assert "data" in result
        assert len(result["data"]) == 1

    @responses.activate
    def test_query_opencost_error(self, enricher):
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            status=500,
        )

        result = enricher.query_opencost("1d")
        assert result == {}

    @responses.activate
    def test_get_gpu_metrics(self, enricher, mock_prometheus_gpu_util):
        # Mock GPU utilization
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json=mock_prometheus_gpu_util,
            status=200,
        )
        # Mock memory utilization (empty)
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        # Mock temperature (empty)
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )

        metrics = enricher.get_gpu_metrics()
        assert len(metrics) == 3
        assert metrics[0].node == "gpu-node-1"
        assert metrics[0].utilization == 75.0

    @responses.activate
    def test_get_opencost_allocation(self, enricher, mock_opencost_response):
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            json=mock_opencost_response,
            status=200,
        )

        allocations = enricher.get_opencost_allocation()
        assert "ml-platform" in allocations
        assert allocations["ml-platform"] == 25.50

    @responses.activate
    def test_calculate_costs(
        self, enricher, mock_prometheus_gpu_util, mock_opencost_response
    ):
        # Setup all mock responses
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json=mock_prometheus_gpu_util,
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            json=mock_opencost_response,
            status=200,
        )

        summaries = enricher.calculate_costs()
        assert "ml-platform" in summaries
        assert summaries["ml-platform"].gpu_count == 2
        # One GPU is idle (15% < 20%)
        assert summaries["ml-platform"].idle_gpu_hours == 1

    def test_generate_recommendations_idle_gpu(self, enricher):
        summaries = {
            "ml-platform": TeamCostSummary(
                team="ml-platform",
                gpu_cost_daily=100.0,
                gpu_count=4,
                idle_gpu_hours=6,
                avg_utilization=40.0,
            )
        }
        enricher._generate_recommendations(summaries)

        recs = summaries["ml-platform"].recommendations
        assert len(recs) > 0
        idle_rec = next((r for r in recs if r.type == "idle_gpu"), None)
        assert idle_rec is not None
        assert idle_rec.severity == "high"

    def test_generate_recommendations_spot_instance(self, enricher):
        summaries = {
            "ml-platform": TeamCostSummary(
                team="ml-platform",
                gpu_cost_daily=200.0,
                gpu_count=4,
                avg_utilization=60.0,
            )
        }
        enricher._generate_recommendations(summaries)

        recs = summaries["ml-platform"].recommendations
        spot_rec = next((r for r in recs if r.type == "spot_instance"), None)
        assert spot_rec is not None
        assert spot_rec.potential_savings_daily > 0


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestAPIEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"

    def test_ready_endpoint_not_initialized(self, client):
        # Enricher is not initialized in test client by default
        import main

        main.enricher = None

        response = client.get("/ready")
        assert response.status_code == 503

    def test_ready_endpoint_initialized(self, client, config_file):
        import main

        main.enricher = GPUEnricher(
            config_path=config_file,
            prometheus_url="http://localhost:9090",
            opencost_url="http://localhost:9003",
        )

        response = client.get("/ready")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "ready"

    @responses.activate
    def test_cost_summary_endpoint(
        self, client, config_file, mock_prometheus_gpu_util, mock_opencost_response
    ):
        import main

        main.enricher = GPUEnricher(
            config_path=config_file,
            prometheus_url="http://localhost:9090",
            opencost_url="http://localhost:9003",
        )

        # Setup mock responses
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json=mock_prometheus_gpu_util,
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            json=mock_opencost_response,
            status=200,
        )

        response = client.get("/api/v1/costs/summary")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "teams" in data
        assert "totals" in data

    @responses.activate
    def test_recommendations_endpoint(
        self, client, config_file, mock_prometheus_gpu_util, mock_opencost_response
    ):
        import main

        main.enricher = GPUEnricher(
            config_path=config_file,
            prometheus_url="http://localhost:9090",
            opencost_url="http://localhost:9003",
        )

        # Setup mock responses
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json=mock_prometheus_gpu_util,
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            json=mock_opencost_response,
            status=200,
        )

        response = client.get("/api/v1/recommendations")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "recommendations" in data
        assert "total_potential_savings_daily" in data

    @responses.activate
    def test_gpu_utilization_endpoint(
        self, client, config_file, mock_prometheus_gpu_util
    ):
        import main

        main.enricher = GPUEnricher(
            config_path=config_file,
            prometheus_url="http://localhost:9090",
            opencost_url="http://localhost:9003",
        )

        # Setup mock responses
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json=mock_prometheus_gpu_util,
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )
        responses.add(
            responses.GET,
            "http://localhost:9090/api/v1/query",
            json={"status": "success", "data": {"result": []}},
            status=200,
        )

        response = client.get("/api/v1/gpu/utilization")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "gpus" in data
        assert data["count"] == 3


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestIntegration:
    @responses.activate
    def test_full_cost_calculation_flow(
        self, enricher, mock_prometheus_gpu_util, mock_opencost_response
    ):
        """Test the complete flow from metrics collection to recommendations."""
        # Setup all mock responses
        for _ in range(3):  # GPU util, mem util, temp
            responses.add(
                responses.GET,
                "http://localhost:9090/api/v1/query",
                json=mock_prometheus_gpu_util
                if len(responses.calls) == 0
                else {
                    "status": "success",
                    "data": {"result": []},
                },
                status=200,
            )
        responses.add(
            responses.GET,
            "http://localhost:9003/allocation/compute",
            json=mock_opencost_response,
            status=200,
        )

        # Calculate costs
        summaries = enricher.calculate_costs()

        # Verify we have team data
        assert len(summaries) > 0

        # Get recommendations
        recs = enricher.get_recommendations()
        assert isinstance(recs, list)

        # Get summary
        summary = enricher.get_cost_summary()
        assert "totals" in summary
        assert "teams" in summary


# =============================================================================
# Config Module Tests
# =============================================================================


class TestConfigModule:
    def test_thresholds_default_values(self):
        from config import ThresholdConfig

        config = ThresholdConfig()
        assert config.idle_gpu_threshold == 20.0
        assert config.underutilized_threshold == 30.0
        assert config.zscore_threshold == 2.5
        assert config.default_rate_limit == 100

    def test_load_thresholds(self):
        from config import load_thresholds

        thresholds = load_thresholds()
        assert thresholds.idle_gpu_threshold >= 0
        assert thresholds.underutilized_threshold >= 0

    def test_validate_team_name_valid(self):
        from config import validate_team_name

        assert validate_team_name("ml-platform") == "ml-platform"
        assert validate_team_name("DataScience") == "datascience"
        assert validate_team_name("team_123") == "team_123"

    def test_validate_team_name_invalid(self):
        from config import validate_team_name, ValidationError

        with pytest.raises(ValidationError) as exc_info:
            validate_team_name("")
        assert exc_info.value.field == "team"

        with pytest.raises(ValidationError):
            validate_team_name("a" * 100)  # Too long

    def test_validate_severity_valid(self):
        from config import validate_severity

        assert validate_severity("low") == "low"
        assert validate_severity("HIGH") == "high"
        assert validate_severity("medium") == "medium"

    def test_validate_severity_invalid(self):
        from config import validate_severity, ValidationError

        with pytest.raises(ValidationError) as exc_info:
            validate_severity("invalid")
        assert exc_info.value.field == "severity"

    def test_validate_period_valid(self):
        from config import validate_period

        assert validate_period("1d") == "1d"
        assert validate_period("7d") == "7d"
        assert validate_period("30d") == "30d"

    def test_validate_period_invalid(self):
        from config import validate_period, ValidationError

        with pytest.raises(ValidationError) as exc_info:
            validate_period("5d")
        assert exc_info.value.field == "period"

    def test_validate_days(self):
        from config import validate_days, ValidationError

        assert validate_days(1) == 1
        assert validate_days(30) == 30
        assert validate_days(365) == 365

        with pytest.raises(ValidationError):
            validate_days(0)

        with pytest.raises(ValidationError):
            validate_days(1000)

    def test_validate_report_format(self):
        from config import validate_report_format, ValidationError

        assert validate_report_format("csv") == "csv"
        assert validate_report_format("PDF") == "pdf"
        assert validate_report_format("json") == "json"

        with pytest.raises(ValidationError):
            validate_report_format("xml")


# =============================================================================
# Auth Module Tests
# =============================================================================


class TestAuthModule:
    def test_api_key_creation(self):
        from auth import APIKey

        key = APIKey(
            key_hash="abc123",
            name="test-key",
            scopes=["read", "write"],
            team="ml-platform",
            rate_limit=100,
        )
        assert key.name == "test-key"
        assert "read" in key.scopes
        assert key.rate_limit == 100

    def test_api_key_scope_membership(self):
        from auth import APIKey

        key = APIKey(
            key_hash="abc123",
            name="test",
            scopes=["read"],
        )
        assert "read" in key.scopes
        assert "write" not in key.scopes

    def test_api_key_defaults(self):
        from auth import APIKey

        key = APIKey(key_hash="abc", name="test", scopes=["read"])
        assert key.team is None
        assert key.rate_limit == 100
        assert key.enabled is True


# =============================================================================
# Budget Forecast Tests
# =============================================================================


class TestBudgetForecast:
    def test_budget_forecast_calculation(self):
        from main import calculate_budget_forecast, TeamBudget

        budget = TeamBudget(
            team="ml-platform",
            monthly_budget=5000.0,
            alert_threshold_pct=80.0,
            critical_threshold_pct=95.0,
        )

        forecast = calculate_budget_forecast(
            team="ml-platform",
            daily_cost=100.0,
            budget=budget,
        )

        assert forecast.team == "ml-platform"
        assert forecast.monthly_budget == 5000.0
        assert forecast.daily_avg_spend > 0
        assert forecast.status in ["on_track", "warning", "critical", "exceeded"]
        assert forecast.trend in ["increasing", "stable", "decreasing"]

    def test_budget_forecast_exceeded(self):
        from main import calculate_budget_forecast, TeamBudget

        budget = TeamBudget(
            team="test-team",
            monthly_budget=100.0,  # Very low budget
        )

        forecast = calculate_budget_forecast(
            team="test-team",
            daily_cost=500.0,  # Very high daily cost
            budget=budget,
        )

        # Should be in warning or critical state
        assert forecast.status in ["warning", "critical", "exceeded"]


# =============================================================================
# API Error Response Tests
# =============================================================================


class TestAPIErrorResponses:
    def test_api_error_function(self, client):
        """Test api_error within app context."""
        from main import api_error, app

        with app.app_context():
            response, status_code = api_error("Test error", 400)
            assert status_code == 400
            assert b"Test error" in response.data

    def test_api_error_with_details(self, client):
        """Test api_error with details within app context."""
        from main import api_error, app

        with app.app_context():
            response, status_code = api_error(
                "Validation failed",
                400,
                details={"field": "team"},
            )
            data = json.loads(response.data)
            assert data["details"]["field"] == "team"


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    def test_module_lock_exists(self):
        from main import _module_lock, _enricher_lock
        import threading

        assert isinstance(_module_lock, type(threading.Lock()))
        assert isinstance(_enricher_lock, type(threading.Lock()))

    def test_lazy_loading_functions_exist(self):
        from main import (
            get_anomaly_detector,
            get_rightsizing_engine,
            get_billing_integration,
        )

        # These functions should exist and be callable
        assert callable(get_anomaly_detector)
        assert callable(get_rightsizing_engine)
        assert callable(get_billing_integration)


# =============================================================================
# Phase 3: Anomaly Detection Tests
# =============================================================================


class TestAnomalyDetection:
    def test_anomaly_detector_initialization(self):
        from anomaly import AnomalyDetector

        detector = AnomalyDetector(zscore_threshold=3.0, iqr_multiplier=2.0)
        assert detector.zscore_threshold == 3.0
        assert detector.iqr_multiplier == 2.0

    def test_zscore_anomaly_detection(self):
        from anomaly import AnomalyDetector

        detector = AnomalyDetector(zscore_threshold=2.0)
        values = [10, 11, 10, 12, 11, 10, 11]

        # Normal value - should not be anomaly
        is_anomaly, z_score, expected = detector.detect_zscore_anomaly(values, 11)
        assert not is_anomaly

        # Extreme value - should be anomaly
        is_anomaly, z_score, expected = detector.detect_zscore_anomaly(values, 100)
        assert is_anomaly
        assert z_score > 2.0

    def test_zscore_insufficient_data(self):
        from anomaly import AnomalyDetector

        detector = AnomalyDetector()
        values = [10, 11]  # Only 2 data points

        is_anomaly, z_score, expected = detector.detect_zscore_anomaly(values, 50)
        assert not is_anomaly  # Not enough data

    def test_iqr_anomaly_detection(self):
        from anomaly import AnomalyDetector

        detector = AnomalyDetector(iqr_multiplier=1.5)
        values = [10, 12, 11, 13, 10, 11, 12, 11]

        # Normal value - should not be anomaly
        is_anomaly, deviation, median = detector.detect_iqr_anomaly(values, 11)
        assert not is_anomaly

        # Outlier - should be anomaly
        is_anomaly, deviation, median = detector.detect_iqr_anomaly(values, 50)
        assert is_anomaly

    def test_moving_average_anomaly(self):
        from anomaly import AnomalyDetector

        detector = AnomalyDetector()
        values = [100] * 30  # Stable at 100

        # 50% spike
        is_anomaly, deviation_pct, moving_avg = detector.detect_moving_average_anomaly(
            values, 150, window_size=24, threshold_pct=30.0
        )
        assert is_anomaly
        assert deviation_pct == 50.0

    def test_cost_anomaly_detection(self):
        from anomaly import AnomalyDetector, AnomalyType

        detector = AnomalyDetector()
        historical_costs = [100, 105, 98, 102, 100, 103, 99, 101]

        # Cost spike
        anomaly = detector.detect_cost_anomaly("test-team", 250, historical_costs)
        assert anomaly is not None
        assert anomaly.type == AnomalyType.COST_SPIKE

    def test_temperature_anomaly_detection(self):
        from anomaly import AnomalyDetector, AnomalyType

        detector = AnomalyDetector()

        # High temperature anomaly
        anomaly = detector.detect_temperature_anomaly(
            node="gpu-node-1",
            gpu_id="0",
            team="test-team",
            current_temp=92,
            utilization=80,
        )
        assert anomaly is not None
        assert anomaly.type == AnomalyType.TEMPERATURE_ANOMALY
        assert anomaly.severity.value == "critical"

    def test_idle_pattern_detection(self):
        from anomaly import AnomalyDetector, AnomalyType

        detector = AnomalyDetector()
        # 6 consecutive low utilization samples
        historical_utils = [15, 12, 10, 8, 14, 11]

        anomaly = detector.detect_utilization_anomaly(
            node="gpu-node-1",
            gpu_id="0",
            team="test-team",
            current_util=10,
            historical_utils=historical_utils,
        )
        assert anomaly is not None
        assert anomaly.type == AnomalyType.IDLE_PATTERN

    def test_run_all_detections(self):
        from anomaly import AnomalyDetector

        detector = AnomalyDetector()
        metrics = {
            "current_cost": 100,
            "historical_costs": [50, 52, 48, 51, 49],
            "gpus": [
                {
                    "node": "gpu-node-1",
                    "gpu_id": "0",
                    "utilization": 10,
                    "temperature": 95,
                }
            ],
            "efficiency": 30,
            "historical_efficiency": [],
        }

        anomalies = detector.run_all_detections("test-team", metrics)
        assert isinstance(anomalies, list)

    def test_anomaly_to_dict(self):
        from anomaly import Anomaly, AnomalyType, AnomalySeverity
        from datetime import datetime, timezone

        anomaly = Anomaly(
            type=AnomalyType.COST_SPIKE,
            severity=AnomalySeverity.WARNING,
            metric_name="daily_cost",
            current_value=150.0,
            expected_value=100.0,
            deviation_pct=50.0,
            team="test-team",
            resource="team_budget",
            timestamp=datetime.now(timezone.utc),
            description="Test anomaly",
            recommendation="Test recommendation",
        )

        result = anomaly.to_dict()
        assert result["type"] == "cost_spike"
        assert result["severity"] == "warning"
        assert result["deviation_pct"] == 50.0


# =============================================================================
# Phase 3: Right-Sizing Tests
# =============================================================================


class TestRightsizing:
    def test_rightsizing_engine_initialization(self):
        from rightsizing import RightsizingEngine, INSTANCE_CATALOG

        engine = RightsizingEngine()
        assert engine.catalog == INSTANCE_CATALOG

    def test_instance_spec_dataclass(self):
        from rightsizing import InstanceSpec

        spec = InstanceSpec(
            instance_type="g4dn.xlarge",
            cloud="aws",
            gpu_count=1,
            gpu_model="T4",
            gpu_memory_gb=16,
            vcpus=4,
            memory_gb=16,
            on_demand_hourly=0.526,
            spot_hourly=0.158,
        )
        assert spec.instance_type == "g4dn.xlarge"
        assert spec.on_demand_hourly == 0.526

    def test_find_smaller_instance(self):
        from rightsizing import RightsizingEngine

        engine = RightsizingEngine()
        smaller = engine.find_smaller_instance("aws:g4dn.2xlarge")
        assert smaller == "aws:g4dn.xlarge"

    def test_find_larger_instance(self):
        from rightsizing import RightsizingEngine

        engine = RightsizingEngine()
        larger = engine.find_larger_instance("aws:g4dn.xlarge")
        assert larger == "aws:g4dn.2xlarge"

    def test_add_utilization_sample(self):
        from rightsizing import RightsizingEngine

        engine = RightsizingEngine()
        engine.add_utilization_sample("node-1", "0", 75.0, 60.0)
        engine.add_utilization_sample("node-1", "0", 80.0, 65.0)

        avg_util, peak_util, avg_mem, count = engine.get_utilization_stats("node-1", "0")
        assert count == 2
        assert avg_util == 77.5
        assert peak_util == 80.0

    def test_analyze_gpu_insufficient_data(self):
        from rightsizing import RightsizingEngine

        engine = RightsizingEngine()
        # Only add a few samples (less than MIN_DATA_POINTS)
        for i in range(5):
            engine.add_utilization_sample("node-1", "0", 50.0, 40.0)

        rec = engine.analyze_gpu("node-1", "0", "test-team", "aws:g4dn.xlarge")
        assert rec is None  # Not enough data

    def test_analyze_gpu_idle_termination(self):
        from rightsizing import RightsizingEngine, RightsizeAction

        engine = RightsizingEngine()
        # Add enough samples with very low utilization
        for i in range(30):
            engine.add_utilization_sample("node-1", "0", 5.0, 5.0)

        rec = engine.analyze_gpu("node-1", "0", "test-team", "aws:g4dn.xlarge")
        assert rec is not None
        assert rec.action == RightsizeAction.TERMINATE
        assert rec.savings_pct == 100

    def test_analyze_gpu_downsize(self):
        from rightsizing import RightsizingEngine, RightsizeAction

        engine = RightsizingEngine()
        # Add enough samples with low utilization
        for i in range(30):
            engine.add_utilization_sample("node-1", "0", 25.0, 30.0)

        rec = engine.analyze_gpu("node-1", "0", "test-team", "aws:g4dn.2xlarge")
        assert rec is not None
        assert rec.action == RightsizeAction.DOWNSIZE
        assert rec.savings_daily > 0

    def test_analyze_gpu_upsize(self):
        from rightsizing import RightsizingEngine, RightsizeAction

        engine = RightsizingEngine()
        # Add enough samples with high utilization
        for i in range(30):
            engine.add_utilization_sample("node-1", "0", 90.0, 85.0)

        rec = engine.analyze_gpu("node-1", "0", "test-team", "aws:g4dn.xlarge")
        assert rec is not None
        assert rec.action == RightsizeAction.UPSIZE
        assert rec.savings_daily < 0  # Negative savings (cost increase)

    def test_analyze_gpu_spot_migration(self):
        from rightsizing import RightsizingEngine, RightsizeAction

        engine = RightsizingEngine()
        # Add enough samples with moderate utilization
        for i in range(30):
            engine.add_utilization_sample("node-1", "0", 50.0, 40.0)

        rec = engine.analyze_gpu(
            "node-1", "0", "test-team", "aws:g4dn.xlarge", is_spot_tolerant=True
        )
        assert rec is not None
        assert rec.action == RightsizeAction.SPOT_MIGRATE
        assert rec.savings_daily > 0

    def test_analyze_team(self):
        from rightsizing import RightsizingEngine

        engine = RightsizingEngine()
        # Pre-populate with enough data
        for i in range(30):
            engine.add_utilization_sample("node-1", "0", 20.0, 15.0)
            engine.add_utilization_sample("node-2", "0", 90.0, 85.0)

        gpus = [
            {
                "node": "node-1",
                "gpu_id": "0",
                "instance_type": "aws:g4dn.xlarge",
                "utilization": 20.0,
                "memory_util": 15.0,
            },
            {
                "node": "node-2",
                "gpu_id": "0",
                "instance_type": "aws:g4dn.xlarge",
                "utilization": 90.0,
                "memory_util": 85.0,
            },
        ]

        recommendations = engine.analyze_team("test-team", gpus)
        assert len(recommendations) >= 1

    def test_rightsizing_recommendation_to_dict(self):
        from rightsizing import RightsizeRecommendation, RightsizeAction
        from datetime import datetime, timezone

        rec = RightsizeRecommendation(
            action=RightsizeAction.DOWNSIZE,
            current_instance="aws:g4dn.2xlarge",
            recommended_instance="g4dn.xlarge",
            team="test-team",
            node="node-1",
            reason="Underutilized",
            current_cost_daily=18.0,
            projected_cost_daily=12.6,
            savings_daily=5.4,
            savings_pct=30.0,
            confidence=0.9,
            utilization_avg=25.0,
            utilization_peak=40.0,
            memory_util_avg=30.0,
            timestamp=datetime.now(timezone.utc),
        )

        result = rec.to_dict()
        assert result["action"] == "downsize"
        assert result["savings_daily"] == 5.4

    def test_get_summary(self):
        from rightsizing import RightsizingEngine, RightsizeRecommendation, RightsizeAction
        from datetime import datetime, timezone

        engine = RightsizingEngine()
        recommendations = [
            RightsizeRecommendation(
                action=RightsizeAction.DOWNSIZE,
                current_instance="g4dn.2xlarge",
                recommended_instance="g4dn.xlarge",
                team="test",
                node="node-1",
                reason="Test",
                current_cost_daily=18.0,
                projected_cost_daily=12.6,
                savings_daily=5.4,
                savings_pct=30.0,
                confidence=0.9,
                utilization_avg=25.0,
                utilization_peak=40.0,
                memory_util_avg=30.0,
                timestamp=datetime.now(timezone.utc),
            )
        ]

        summary = engine.get_summary(recommendations)
        assert summary["total_recommendations"] == 1
        assert summary["potential_savings_daily"] == 5.4
        assert summary["potential_savings_monthly"] == 162.0


# =============================================================================
# Phase 3: Report Generation Tests
# =============================================================================


class TestReportGeneration:
    def test_report_metadata(self):
        from reports import ReportMetadata, ReportPeriod
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        metadata = ReportMetadata(
            title="Test Report",
            period=ReportPeriod.MONTHLY,
            start_date=now,
            end_date=now,
            generated_at=now,
            generated_by="test",
        )
        assert metadata.title == "Test Report"
        assert metadata.period == ReportPeriod.MONTHLY

    def test_team_cost_report(self):
        from reports import TeamCostReport

        report = TeamCostReport(
            team="test-team",
            gpu_cost=100.0,
            k8s_cost=50.0,
            total_cost=150.0,
            gpu_hours=24.0,
            gpu_count=1,
            avg_utilization=75.0,
            idle_hours=2.0,
            spot_savings_potential=30.0,
            budget=5000.0,
            budget_remaining=4850.0,
            cost_trend_pct=5.0,
        )
        assert report.total_cost == 150.0

    def test_generate_csv_report(self):
        from reports import ChargebackReportGenerator, TeamCostReport, ReportMetadata, ReportPeriod
        from datetime import datetime, timezone

        generator = ChargebackReportGenerator()
        now = datetime.now(timezone.utc)

        metadata = ReportMetadata(
            title="Test Report",
            period=ReportPeriod.MONTHLY,
            start_date=now,
            end_date=now,
            generated_at=now,
            generated_by="test",
        )

        team_reports = [
            TeamCostReport(
                team="ml-platform",
                gpu_cost=500.0,
                k8s_cost=200.0,
                total_cost=700.0,
                gpu_hours=720.0,
                gpu_count=1,
                avg_utilization=75.0,
                idle_hours=48.0,
                spot_savings_potential=150.0,
                budget=5000.0,
                budget_remaining=4300.0,
                cost_trend_pct=10.0,
            )
        ]

        csv_content = generator.generate_csv(team_reports, metadata)
        assert "AI FinOps Chargeback Report" in csv_content
        assert "ml-platform" in csv_content
        assert "700.00" in csv_content

    def test_generate_json_report(self):
        from reports import ChargebackReportGenerator, TeamCostReport, ReportMetadata, ReportPeriod, ReportFormat
        from datetime import datetime, timezone
        import json

        generator = ChargebackReportGenerator()
        now = datetime.now(timezone.utc)

        metadata = ReportMetadata(
            title="Test Report",
            period=ReportPeriod.MONTHLY,
            start_date=now,
            end_date=now,
            generated_at=now,
            generated_by="test",
        )

        team_reports = [
            TeamCostReport(
                team="ml-platform",
                gpu_cost=500.0,
                k8s_cost=200.0,
                total_cost=700.0,
                gpu_hours=720.0,
                gpu_count=1,
                avg_utilization=75.0,
                idle_hours=48.0,
                spot_savings_potential=150.0,
                budget=5000.0,
                budget_remaining=4300.0,
                cost_trend_pct=10.0,
            )
        ]

        content, content_type = generator.generate_report(
            team_reports, metadata, format=ReportFormat.JSON
        )

        assert content_type == "application/json"
        data = json.loads(content.decode("utf-8"))
        assert data["summary"]["total_cost"] == 700.0
        assert len(data["teams"]) == 1

    def test_generate_monthly_chargeback_report(self):
        from reports import generate_monthly_chargeback_report

        team_data = {
            "ml-platform": {
                "gpu_cost_daily": 50.0,
                "k8s_cost_daily": 20.0,
                "total_cost_daily": 70.0,
                "gpu_count": 2,
                "avg_utilization": 65.0,
                "idle_gpu_hours": 4.0,
            }
        }

        content, content_type, filename = generate_monthly_chargeback_report(
            team_data=team_data,
            recommendations=[],
            anomalies=[],
            format="csv",
        )

        assert content_type == "text/csv"
        assert "chargeback-report" in filename
        assert b"ml-platform" in content


# =============================================================================
# Phase 3: Billing Integration Tests
# =============================================================================


class TestBillingIntegration:
    def test_billing_record_dataclass(self):
        from billing import BillingRecord
        from datetime import datetime, timezone

        record = BillingRecord(
            date=datetime.now(timezone.utc),
            service="EC2",
            resource_id="i-1234567890",
            resource_tags={"team": "ml-platform"},
            usage_type="BoxUsage:g4dn.xlarge",
            usage_quantity=24.0,
            usage_unit="Hours",
            cost=12.62,
            currency="USD",
            cloud_provider="aws",
        )
        assert record.cost == 12.62
        assert record.cloud_provider == "aws"

    def test_team_billing_data_to_dict(self):
        from billing import TeamBillingData
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        data = TeamBillingData(
            team="ml-platform",
            period_start=now,
            period_end=now,
            total_cost=1000.0,
            gpu_cost=800.0,
            compute_cost=100.0,
            storage_cost=50.0,
            network_cost=30.0,
            other_cost=20.0,
            currency="USD",
            records=[],
        )

        result = data.to_dict()
        assert result["team"] == "ml-platform"
        assert result["total_cost"] == 1000.0
        assert result["breakdown"]["gpu"] == 800.0

    def test_billing_integration_disabled_by_default(self):
        from billing import BillingIntegration

        integration = BillingIntegration()
        assert not integration.is_enabled
        assert integration.enabled_providers == []

    def test_get_actual_costs_when_disabled(self):
        from billing import BillingIntegration

        integration = BillingIntegration()
        result = integration.get_actual_costs()
        assert result == {}

    def test_get_cost_comparison_no_data(self):
        from billing import BillingIntegration

        integration = BillingIntegration()
        result = integration.get_cost_comparison({"team1": 100.0})
        assert result["status"] == "no_data"


# =============================================================================
# Phase 3: Config Updates Tests
# =============================================================================


class TestConfigUpdates:
    def test_min_cost_factor_config(self):
        from config import ThresholdConfig

        config = ThresholdConfig()
        assert config.min_cost_factor == 0.0

    def test_default_team_budget_config(self):
        from config import ThresholdConfig

        config = ThresholdConfig()
        assert config.default_team_budget == 5000.0

    def test_default_gpu_pricing_config(self):
        from config import DEFAULT_GPU_PRICING

        assert "aws" in DEFAULT_GPU_PRICING
        assert "gcp" in DEFAULT_GPU_PRICING
        assert "azure" in DEFAULT_GPU_PRICING
        assert "g4dn.xlarge" in DEFAULT_GPU_PRICING["aws"]
        assert DEFAULT_GPU_PRICING["aws"]["g4dn.xlarge"]["on_demand"] == 0.526


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
