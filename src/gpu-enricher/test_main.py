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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
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
    def test_calculate_costs(self, enricher, mock_prometheus_gpu_util, mock_opencost_response):
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
    def test_cost_summary_endpoint(self, client, config_file, mock_prometheus_gpu_util, mock_opencost_response):
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
    def test_recommendations_endpoint(self, client, config_file, mock_prometheus_gpu_util, mock_opencost_response):
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
    def test_gpu_utilization_endpoint(self, client, config_file, mock_prometheus_gpu_util):
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
    def test_full_cost_calculation_flow(self, enricher, mock_prometheus_gpu_util, mock_opencost_response):
        """Test the complete flow from metrics collection to recommendations."""
        # Setup all mock responses
        for _ in range(3):  # GPU util, mem util, temp
            responses.add(
                responses.GET,
                "http://localhost:9090/api/v1/query",
                json=mock_prometheus_gpu_util if len(responses.calls) == 0 else {
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
