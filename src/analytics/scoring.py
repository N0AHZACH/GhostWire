"""
GhostWire Analytics Layer

Responsible for:
- Hallucination metrics
- Calibration analysis
- Reliability scoring
- Risk grading
- Report generation
- Session handshake with Auditor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

try:
    from src.core.auditor import GhostwireAuditor  # type: ignore
except ImportError:
    GhostwireAuditor = Any

logger = logging.getLogger(__name__)

@dataclass
class AuditResult:
    """Matches the JSON structure provided by the Pipeline Architect's Engine"""
    prompt: str
    response: str
    is_hallucination: bool
    confidence: int
    risk_level: int
    explanation: str = ""

    def to_dict(self):
        return asdict(self)

class HallucinationScorer:
    """
    Static analytics layer for GhostWire.
    Computes reliability and risk metrics from audit results.
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
        Calibration Gap = | (avg_confidence / 100) - accuracy |
        """
        if not results:
            return 0.0

        accuracy = HallucinationScorer.calculate_accuracy(results)
        avg_conf = HallucinationScorer.calculate_average_confidence(results) / 100

        return abs(avg_conf - accuracy)

    # ---------------------------------------------------------------------
    # Risk Distribution
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_risk_distribution(results: List[AuditResult]) -> Dict[int, int]:
        distribution = {i: 0 for i in range(1, 6)}

        for r in results:
            level = max(1, min(5, r.risk_level))
            distribution[level] += 1

        return distribution

    # ---------------------------------------------------------------------
    # Reliability Score
    # ---------------------------------------------------------------------

    @staticmethod
    def calculate_reliability_score(results: List[AuditResult]) -> float:
        """
        reliability = accuracy * (1 - calibration_gap)
        Returns value between 0 and 1.
        """
        if not results:
            return 0.0

        accuracy = HallucinationScorer.calculate_accuracy(results)
        gap = HallucinationScorer.calculate_calibration_gap(results)

        score = accuracy * (1 - gap)
        return max(0.0, min(1.0, score))

    @staticmethod
    def assign_risk_grade(score: float) -> str:
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
    # Full Report (Detailed Analytics)
    # ---------------------------------------------------------------------

    @staticmethod
    def generate_report(results: List[AuditResult]) -> Dict[str, Any]:
        """
        Full analytics report used internally or by dashboard.
        """

        total = len(results)
        hallucination_count = sum(1 for r in results if r.is_hallucination)

        accuracy = HallucinationScorer.calculate_accuracy(results)
        hallucination_rate = HallucinationScorer.calculate_hallucination_rate(results)
        avg_confidence = HallucinationScorer.calculate_average_confidence(results)
        calibration_gap = HallucinationScorer.calculate_calibration_gap(results)
        reliability_score = HallucinationScorer.calculate_reliability_score(results)
        risk_grade = HallucinationScorer.assign_risk_grade(reliability_score)
        risk_distribution = HallucinationScorer.calculate_risk_distribution(results)

        report = {
            "total_audits": total,
            "hallucination_count": hallucination_count,
            "hallucination_rate": hallucination_rate,
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
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

        logger.info("Full analytics report generated — %d audits.", total)
        return report

    # ---------------------------------------------------------------------
    # SESSION HANDSHAKE WITH AUDITOR
    # ---------------------------------------------------------------------

    @staticmethod
    def build_session_metrics(results: List[AuditResult]) -> Dict[str, Any]:
        """
        Builds compact session-level metrics required by Auditor.

        REQUIRED KEYS:
            total_runs
            reliability_score
            avg_confidence
            critical_count
        """

        total_runs = len(results)
        reliability_score = HallucinationScorer.calculate_reliability_score(results)
        avg_confidence = HallucinationScorer.calculate_average_confidence(results)

        critical_count = sum(
            1 for r in results
            if r.is_hallucination and r.risk_level >= 4
        )

        session_metrics = {
            "total_runs": total_runs,
            "reliability_score": reliability_score,
            "avg_confidence": avg_confidence,
            "critical_count": critical_count,
        }

        logger.info("Session metrics built for Auditor.")
        return session_metrics

    @staticmethod
    def send_to_auditor(
        results: List[AuditResult],
        auditor: GhostwireAuditor,
    ) -> Any:
        """
        Builds session metrics and sends them to Auditor.
        """

        session_metrics = HallucinationScorer.build_session_metrics(results)

        # HANDSHAKE: pass dictionary to Auditor
        return auditor.generate_final_report(session_metrics)
