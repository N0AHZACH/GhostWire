"""
Hallucination scoring and reliability analytics for GhostWire.

Provides:
- Core metrics (accuracy, hallucination rate, confidence)
- Calibration analysis
- Reliability scoring
- Risk grading
- Structured report generation
"""

from __future__ import annotations

from typing import List, Dict, Any
import logging

from src.core.engine import AuditResult

logger = logging.getLogger(__name__)


class HallucinationScorer:
    """
    Static analytics layer for evaluating hallucination audits.
    """

    # ---------------------------------------------------------------------
    # Core Metrics
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_hallucination_rate(results: List[AuditResult]) -> float:
        if not results:
            return 0.0
        hallucinations = sum(1 for r in results if r.is_hallucination)
        return hallucinations / len(results)

    @staticmethod
    def calculate_accuracy(results: List[AuditResult]) -> float:
        if not results:
            return 0.0
        correct = sum(1 for r in results if not r.is_hallucination)
        return correct / len(results)

    @staticmethod
    def calculate_average_confidence(results: List[AuditResult]) -> float:
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)

    # ---------------------------------------------------------------------
    # Calibration
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_calibration_gap(results: List[AuditResult]) -> float:
        """
        Measures how well model confidence aligns with reality.

        Gap = | (average_confidence / 100) - accuracy |
        """
        if not results:
            return 0.0

        accuracy = HallucinationScorer.calculate_accuracy(results)
        avg_confidence = (
            HallucinationScorer.calculate_average_confidence(results) / 100
        )

        return abs(avg_confidence - accuracy)

    # ---------------------------------------------------------------------
    # Risk Distribution
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_risk_distribution(
        results: List[AuditResult],
    ) -> Dict[int, int]:
        """
        Returns a count of risk levels (1–5).
        """
        distribution = {i: 0 for i in range(1, 6)}

        for r in results:
            level = max(1, min(5, r.risk_level))
            distribution[level] += 1

        return distribution

    # ---------------------------------------------------------------------
    # Reliability Scoring
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_reliability_score(results: List[AuditResult]) -> float:
        """
        Composite trust score.

        Formula:
            reliability = accuracy * (1 - calibration_gap)

        Returns value between 0 and 1.
        """
        if not results:
            return 0.0

        accuracy = HallucinationScorer.calculate_accuracy(results)
        calibration_gap = HallucinationScorer.calculate_calibration_gap(results)

        score = accuracy * (1 - calibration_gap)
        return max(0.0, min(1.0, score))

    @staticmethod
    def assign_risk_grade(score: float) -> str:
        """
        Converts reliability score into A–F grade.
        """
        if score >= 0.90:
            return "A"
        elif score >= 0.75:
            return "B"
        elif score >= 0.60:
            return "C"
        elif score >= 0.40:
            return "D"
        else:
            return "F"

    # ---------------------------------------------------------------------
    # Report Generation
    # ---------------------------------------------------------------------

    @staticmethod
    def generate_report(results: List[AuditResult]) -> Dict[str, Any]:
        """
        Produce a summary report dictionary suitable for JSON serialization.

        Returns:
        {
            "total_audits": int,
            "hallucination_count": int,
            "hallucination_rate": float,
            "accuracy": float,
            "average_confidence": float,
            "calibration_gap": float,
            "reliability_score": float,
            "risk_grade": str,
            "risk_distribution": dict,
            "high_risk_items": list[dict],
        }
        """
        total = len(results)
        hallucination_count = sum(1 for r in results if r.is_hallucination)

        accuracy = HallucinationScorer.calculate_accuracy(results)
        hallucination_rate = HallucinationScorer.calculate_hallucination_rate(results)
        average_confidence = HallucinationScorer.calculate_average_confidence(results)
        calibration_gap = HallucinationScorer.calculate_calibration_gap(results)
        reliability_score = HallucinationScorer.calculate_reliability_score(results)
        risk_grade = HallucinationScorer.assign_risk_grade(reliability_score)
        risk_distribution = HallucinationScorer.calculate_risk_distribution(results)

        report: Dict[str, Any] = {
            "total_audits": total,
            "hallucination_count": hallucination_count,
            "hallucination_rate": hallucination_rate,
            "accuracy": accuracy,
            "average_confidence": average_confidence,
            "calibration_gap": calibration_gap,
            "reliability_score": reliability_score,
            "risk_grade": risk_grade,
            "risk_distribution": risk_distribution,
            "high_risk_items": [
                r.to_dict()
                for r in results
                if r.is_hallucination and r.risk_level >= 4
            ],
        }

        logger.info("Reliability report generated — %d audits.", total)
        return report
