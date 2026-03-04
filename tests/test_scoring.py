"""
Unit tests for HallucinationScorer and reporting logic.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.analytics.scoring import HallucinationScorer, AuditResult


# ---------------------------------------------------------------------------
# Test Data Helpers
# ---------------------------------------------------------------------------

def make_result(
    *,
    is_hallucination: bool,
    confidence: int,
    risk_level: int,
) -> AuditResult:
    """Create a minimal AuditResult for scoring tests."""
    return AuditResult(
        prompt="Q",
        response="A",
        is_hallucination=is_hallucination,
        explanation="Test",
        confidence=confidence,
        risk_level=risk_level,
    )


@pytest.fixture
def sample_results():
    """
    5 results:
    - 2 hallucinations
    - 3 correct
    """
    return [
        make_result(is_hallucination=True, confidence=90, risk_level=4),
        make_result(is_hallucination=True, confidence=80, risk_level=5),
        make_result(is_hallucination=False, confidence=85, risk_level=2),
        make_result(is_hallucination=False, confidence=70, risk_level=1),
        make_result(is_hallucination=False, confidence=60, risk_level=3),
    ]


# ---------------------------------------------------------------------------
# Tests — Core Metrics
# ---------------------------------------------------------------------------

class TestCoreMetrics:

    def test_hallucination_rate(self, sample_results):
        rate = HallucinationScorer.calculate_hallucination_rate(sample_results)
        assert rate == pytest.approx(2 / 5)

    def test_accuracy(self, sample_results):
        accuracy = HallucinationScorer.calculate_accuracy(sample_results)
        assert accuracy == pytest.approx(3 / 5)

    def test_average_confidence(self, sample_results):
        avg = HallucinationScorer.calculate_average_confidence(sample_results)
        expected = (90 + 80 + 85 + 70 + 60) / 5
        assert avg == pytest.approx(expected)

    def test_calibration_gap(self, sample_results):
        """
        Calibration gap = |average_confidence/100 - accuracy|
        """
        accuracy = 3 / 5
        avg_conf = (90 + 80 + 85 + 70 + 60) / 5 / 100
        expected_gap = abs(avg_conf - accuracy)

        gap = HallucinationScorer.calculate_calibration_gap(sample_results)
        assert gap == pytest.approx(expected_gap)


# ---------------------------------------------------------------------------
# Tests — Risk Distribution
# ---------------------------------------------------------------------------

class TestRiskDistribution:

    def test_risk_distribution_counts(self, sample_results):
        dist = HallucinationScorer.calculate_risk_distribution(sample_results)

        assert dist[1] == 1
        assert dist[2] == 1
        assert dist[3] == 1
        assert dist[4] == 1
        assert dist[5] == 1


# ---------------------------------------------------------------------------
# Tests — Reliability + Grade
# ---------------------------------------------------------------------------

class TestReliability:

    def test_reliability_score_range(self, sample_results):
        score = HallucinationScorer.calculate_reliability_score(sample_results)
        assert 0 <= score <= 1

    def test_assign_risk_grade(self):
        assert HallucinationScorer.assign_risk_grade(0.95) == "A"
        assert HallucinationScorer.assign_risk_grade(0.80) in {"A", "B"}
        assert HallucinationScorer.assign_risk_grade(0.60) in {"B", "C"}
        assert HallucinationScorer.assign_risk_grade(0.30) in {"C", "D", "F"}


# ---------------------------------------------------------------------------
# Tests — generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:

    def test_generate_report_structure(self, sample_results):
        report = HallucinationScorer.generate_report(sample_results)

        expected_keys = {
            "total_audits",
            "hallucination_count",
            "hallucination_rate",
            "accuracy",
            "average_confidence",
            "calibration_gap",
            "reliability_score",
            "risk_grade",
            "risk_distribution",
            "high_risk_items",
        }

        assert set(report.keys()) == expected_keys

    def test_generate_report_values(self, sample_results):
        report = HallucinationScorer.generate_report(sample_results)

        assert report["total_audits"] == 5
        assert report["hallucination_count"] == 2
        assert report["hallucination_rate"] == pytest.approx(2 / 5)
        assert report["accuracy"] == pytest.approx(3 / 5)

    def test_generate_report_high_risk_filter(self, sample_results):
        report = HallucinationScorer.generate_report(sample_results)

        high_risk = report["high_risk_items"]
        assert len(high_risk) == 2
        for item in high_risk:
            assert item["risk_level"] >= 4
            assert item["is_hallucination"] is True

    def test_generate_report_empty(self):
        report = HallucinationScorer.generate_report([])

        assert report["total_audits"] == 0
        assert report["hallucination_rate"] == 0.0
        assert report["accuracy"] == 0.0
        assert report["high_risk_items"] == []


# ---------------------------------------------------------------------------
# Tests — Session Handshake (Metrics → Auditor)
# ---------------------------------------------------------------------------

class TestSessionHandshake:

    def test_build_session_metrics_keys(self, sample_results):
        metrics = HallucinationScorer.build_session_metrics(sample_results)

        expected_keys = {
            "total_runs",
            "reliability_score",
            "avg_confidence",
            "critical_count",
        }

        assert set(metrics.keys()) == expected_keys

    def test_build_session_metrics_values(self, sample_results):
        metrics = HallucinationScorer.build_session_metrics(sample_results)

        assert metrics["total_runs"] == len(sample_results)
        assert 0 <= metrics["reliability_score"] <= 1
        assert metrics["avg_confidence"] > 0
        assert metrics["critical_count"] == 2  # risk_level >= 4 & hallucination

    def test_send_to_auditor_calls_generate_final_report(self, sample_results):
        mock_auditor = MagicMock()

        HallucinationScorer.send_to_auditor(sample_results, mock_auditor)

        mock_auditor.generate_final_report.assert_called_once()

        passed_dict = mock_auditor.generate_final_report.call_args[0][0]

        assert "total_runs" in passed_dict
        assert "reliability_score" in passed_dict
        assert "avg_confidence" in passed_dict
        assert "critical_count" in passed_dict
