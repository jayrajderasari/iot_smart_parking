"""
Penalty Engine Module

Converts detected violations into penalty scores.

Core formula:
    Penalty = α·Duration + β·Severity + γ·Repetition

Where:
    - Duration: Time violation persists (seconds) → fairness
    - Severity: Spatial misalignment score [0,1] → impact
    - Repetition: Behavioral pattern (number of violations) → pattern
    - α, β, γ: Configurable weights
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class ViolationRecord:
    """Record of a single parking violation."""
    slot_id: int
    vehicle_id: Optional[int]
    severity_score: float  # [0, 1] from alignment
    duration_seconds: float  # Time in violation
    start_time: datetime
    end_time: Optional[datetime] = None
    confirmed: bool = False
    temporal_duration_frames: int = 0  # Frames in violation


@dataclass
class PenaltyScoreBreakdown:
    """Detailed breakdown of penalty calculation."""
    total_penalty: float
    duration_component: float
    severity_component: float
    repetition_component: float
    alpha_weight: float
    beta_weight: float
    gamma_weight: float
    violation_count: int
    details: Dict = field(default_factory=dict)


class PenaltyEngine:
    """
    Computes parking violation penalties based on spatial and temporal factors.
    
    Converts alignment violations into actionable penalty scores.
    """

    def __init__(self,
                 alpha: float = 0.3,  # Duration weight
                 beta: float = 0.4,   # Severity weight
                 gamma: float = 0.3,  # Repetition weight
                 max_penalty: float = 100.0):
        """
        Initialize penalty engine with weights.
        
        Args:
            alpha: Duration weight (fairness factor)
            beta: Severity weight (spatial impact)
            gamma: Repetition weight (behavioral pattern)
            max_penalty: Maximum penalty score
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_penalty = max_penalty

        # Normalize weights
        total_weight = alpha + beta + gamma
        self.alpha /= total_weight
        self.beta /= total_weight
        self.gamma /= total_weight

        # Violation history per slot/vehicle
        self.violation_history: Dict[int, List[ViolationRecord]] = {}

    def compute_penalty(self,
                       slot_id: int,
                       severity_score: float,
                       duration_seconds: float,
                       violation_count: int = 1,
                       max_duration_baseline: float = 3600.0) -> PenaltyScoreBreakdown:
        """
        Compute penalty score for a violation.
        
        Penalty = α·Duration + β·Severity + γ·Repetition
        
        Args:
            slot_id: Parking slot ID
            severity_score: Spatial severity [0, 1] (1 - alignment_score)
            duration_seconds: Violation duration in seconds
            violation_count: Number of violations for this slot
            max_duration_baseline: Reference duration for normalization (default: 1 hour)
            
        Returns:
            PenaltyScoreBreakdown: Detailed penalty breakdown
        """
        # Normalize duration component [0, 1]
        # Saturates at max_duration_baseline to avoid extreme penalties
        normalized_duration = min(duration_seconds / max_duration_baseline, 1.0)

        # Severity is already normalized [0, 1]
        normalized_severity = float(np.clip(severity_score, 0, 1))

        # Normalize repetition [0, 1]
        # Saturates at 5 violations (adjust if needed)
        normalized_repetition = min(violation_count / 5.0, 1.0)

        # Compute weighted components
        duration_component = self.alpha * normalized_duration
        severity_component = self.beta * normalized_severity
        repetition_component = self.gamma * normalized_repetition

        # Total penalty [0, 1]
        raw_penalty = duration_component + severity_component + repetition_component

        # Scale to max penalty
        final_penalty = min(raw_penalty * self.max_penalty, self.max_penalty)

        breakdown = PenaltyScoreBreakdown(
            total_penalty=float(final_penalty),
            duration_component=float(duration_component),
            severity_component=float(severity_component),
            repetition_component=float(repetition_component),
            alpha_weight=self.alpha,
            beta_weight=self.beta,
            gamma_weight=self.gamma,
            violation_count=violation_count,
            details={
                "normalized_duration": float(normalized_duration),
                "normalized_severity": float(normalized_severity),
                "normalized_repetition": float(normalized_repetition),
                "raw_penalty": float(raw_penalty),
                "max_duration_baseline": max_duration_baseline
            }
        )

        return breakdown

    def record_violation(self,
                        slot_id: int,
                        severity_score: float,
                        duration_seconds: float,
                        vehicle_id: Optional[int] = None,
                        start_time: Optional[datetime] = None) -> None:
        """
        Record a violation in history.
        
        Args:
            slot_id: Parking slot ID
            severity_score: Spatial severity [0, 1]
            duration_seconds: Duration of violation
            vehicle_id: Optional vehicle ID
            start_time: Optional violation start time
        """
        if start_time is None:
            start_time = datetime.now()

        if slot_id not in self.violation_history:
            self.violation_history[slot_id] = []

        violation = ViolationRecord(
            slot_id=slot_id,
            vehicle_id=vehicle_id,
            severity_score=severity_score,
            duration_seconds=duration_seconds,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=duration_seconds),
            confirmed=True
        )

        self.violation_history[slot_id].append(violation)

    def get_repetition_count(self, slot_id: int, time_window_hours: float = 24) -> int:
        """
        Get number of violations in time window.
        
        Args:
            slot_id: Parking slot ID
            time_window_hours: Time window to consider (default: 24 hours)
            
        Returns:
            int: Number of violations in window
        """
        if slot_id not in self.violation_history:
            return 0

        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_violations = [v for v in self.violation_history[slot_id]
                           if v.start_time > cutoff_time]

        return len(recent_violations)

    def get_violation_statistics(self, slot_id: int) -> Dict:
        """
        Get comprehensive violation statistics for a slot.
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            Dict: Statistics including counts, totals, averages
        """
        if slot_id not in self.violation_history:
            return {
                "slot_id": slot_id,
                "total_violations": 0,
                "total_penalty_time": 0,
                "avg_severity": 0,
                "max_severity": 0
            }

        violations = self.violation_history[slot_id]

        return {
            "slot_id": slot_id,
            "total_violations": len(violations),
            "total_penalty_time_seconds": sum(v.duration_seconds for v in violations),
            "avg_duration_seconds": np.mean([v.duration_seconds for v in violations]),
            "avg_severity": float(np.mean([v.severity_score for v in violations])),
            "max_severity": float(max([v.severity_score for v in violations], default=0)),
            "first_violation": violations[0].start_time if violations else None,
            "last_violation": violations[-1].start_time if violations else None
        }

    def compute_cumulative_penalty(self,
                                  slot_id: int,
                                  time_window_hours: float = 24) -> float:
        """
        Compute cumulative penalty across all violations in time window.
        
        Args:
            slot_id: Parking slot ID
            time_window_hours: Time window to consider
            
        Returns:
            float: Total penalty score
        """
        if slot_id not in self.violation_history:
            return 0.0

        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_violations = [v for v in self.violation_history[slot_id]
                           if v.start_time > cutoff_time]

        if not recent_violations:
            return 0.0

        # Sum penalties for each violation
        total_penalty = 0.0
        for violation in recent_violations:
            breakdown = self.compute_penalty(
                slot_id,
                violation.severity_score,
                violation.duration_seconds,
                len(recent_violations)
            )
            total_penalty += breakdown.total_penalty

        return min(total_penalty, self.max_penalty)

    def generate_penalty_report(self, slot_id: int) -> Dict:
        """
        Generate comprehensive penalty report for a slot.
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            Dict: Complete penalty analysis
        """
        stats = self.get_violation_statistics(slot_id)
        
        if stats["total_violations"] == 0:
            return {
                "slot_id": slot_id,
                "status": "CLEAN",
                "total_penalty": 0.0,
                "violations": 0
            }

        # Compute penalty for latest violation
        violations = self.violation_history.get(slot_id, [])
        latest_violation = violations[-1]

        breakdown = self.compute_penalty(
            slot_id,
            latest_violation.severity_score,
            latest_violation.duration_seconds,
            len(violations)
        )

        return {
            "slot_id": slot_id,
            "status": "VIOLATION",
            "total_violations": stats["total_violations"],
            "penalty_breakdown": {
                "total": breakdown.total_penalty,
                "duration_component": breakdown.duration_component,
                "severity_component": breakdown.severity_component,
                "repetition_component": breakdown.repetition_component
            },
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma
            },
            "statistics": stats
        }

    def clear_history(self) -> None:
        """Clear all violation history."""
        self.violation_history.clear()
