"""
HallucinationScorer — Analytics module for hallucination metrics.

This module is owned by the Metrics & Risk Analysts (Role 5). It consumes
``AuditResult`` objects produced by ``GhostWireEngine`` and computes
aggregate statistics.

Usage:
    from src.analytics.scoring import HallucinationScorer
    from src.core.engine import AuditResult

    scorer = HallucinationScorer()
    rate = scorer.calculate_hallucination_rate(results)
    report = scorer.generate_report(results)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, List, Any

# We import the type only — no runtime dependency on the engine internals.
from src.core.engine import AuditResult

logger = logging.getLogger(__name__)


class HallucinationScorer:
    """
    Computes hallucination metrics from a batch of ``AuditResult`` objects.
    """

    @staticmethod
    def calculate_hallucination_rate(results: List[AuditResult]) -> float:
        """
        Return the hallucination rate as a float in [0.0, 1.0].

        Parameters
        ----------
        results : list[AuditResult]
            Audit results from ``GhostWireEngine.run_audit``.

        Returns
        -------
        float
            Fraction of results flagged as hallucinations.
        """
        if not results:
            return 0.0
        hallucinated = sum(1 for r in results if r.is_hallucination)
        rate = hallucinated / len(results)
        logger.info("Hallucination rate: %.2f%% (%d/%d)", rate * 100, hallucinated, len(results))
        return rate

    @staticmethod
    def calculate_risk_distribution(results: List[AuditResult]) -> Dict[int, int]:
        """
        Group audit results by ``risk_level`` (1‑5).

        Returns
        -------
        dict[int, int]
            Mapping of risk_level → count.
        """
        distribution: Dict[int, int] = Counter(r.risk_level for r in results)
        # Ensure all levels are represented.
        for level in range(1, 6):
            distribution.setdefault(level, 0)
        return dict(sorted(distribution.items()))

    @staticmethod
    def calculate_average_confidence(results: List[AuditResult]) -> float:
        """
        Return the mean confidence score across all audit results.
        """
        if not results:
            return 0.0
        return sum(r.confidence for r in results) / len(results)

    @staticmethod
    def generate_report(results: List[AuditResult]) -> Dict[str, Any]:
        """
        Produce a summary report dictionary suitable for JSON serialization.

        Returns
        -------
        dict
            {
                "total_audits": int,
                "hallucination_count": int,
                "hallucination_rate": float,
                "average_confidence": float,
                "risk_distribution": dict,
                "high_risk_items": list[dict],
            }
        """
        scorer = HallucinationScorer

        hallucination_count = sum(1 for r in results if r.is_hallucination)

        report: Dict[str, Any] = {
            "total_audits": len(results),
            "hallucination_count": hallucination_count,
            "hallucination_rate": scorer.calculate_hallucination_rate(results),
            "average_confidence": scorer.calculate_average_confidence(results),
            "risk_distribution": scorer.calculate_risk_distribution(results),
            "high_risk_items": [
                r.to_dict() for r in results
                if r.is_hallucination and r.risk_level >= 4
            ],
        }

        logger.info("Report generated — %d total audits.", len(results))
        return report
