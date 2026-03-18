"""
Temporal Verification Module

Tracks slot occupancy and violation states over time.
Confirms improper parking violations across multiple frames.

Core principle:
Violation confirmed if: ∑(V_t) ≥ T_threshold
where V_t = 1 if violation detected at time t

This eliminates transient false positives from brief detection errors.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime


@dataclass
class FrameObservation:
    """Single frame observation for a slot."""
    timestamp: datetime
    is_improperly_parked: bool
    alignment_score: float
    severity_score: float
    confidence: float
    vehicle_id: Optional[int] = None


@dataclass
class SlotTemporalState:
    """Temporal state tracking for a parking slot."""
    slot_id: int
    observations: List[FrameObservation] = field(default_factory=list)
    violation_count: int = 0
    violation_confirmed: bool = False
    violation_duration_frames: int = 0
    first_violation_time: Optional[datetime] = None
    last_violation_time: Optional[datetime] = None

    def add_observation(self, observation: FrameObservation) -> None:
        """Add a frame observation."""
        self.observations.append(observation)
        
        if observation.is_improperly_parked:
            self.violation_count += 1
            self.last_violation_time = observation.timestamp
            if self.first_violation_time is None:
                self.first_violation_time = observation.timestamp

    def get_violation_duration(self) -> float:
        """Get duration of violation in seconds."""
        if self.first_violation_time is None or self.last_violation_time is None:
            return 0.0
        return (self.last_violation_time - self.first_violation_time).total_seconds()


class TemporalVerificationModule:
    """
    Tracks parking violations over time to confirm improper parking.
    
    Eliminates false positives from transient detection errors.
    Computes violation duration for penalty calculation.
    """

    def __init__(self, confirmation_threshold: int = 3, window_size: int = 10):
        """
        Initialize temporal verification.
        
        Args:
            confirmation_threshold: Frames required to confirm violation (T)
            window_size: Number of frames to maintain in history
        """
        self.confirmation_threshold = confirmation_threshold
        self.window_size = window_size
        self.slot_states: Dict[int, SlotTemporalState] = {}

    def process_frame_observations(self,
                                   frame_number: int,
                                   slot_observations: Dict[int, Tuple[bool, float, float, float]],
                                   timestamp: datetime = None) -> Dict[int, bool]:
        """
        Process observations from a single frame.
        
        Args:
            frame_number: Frame index
            slot_observations: Dict mapping slot_id to (is_improper, alignment_score, 
                              severity_score, confidence)
            timestamp: Frame timestamp (defaults to now)
            
        Returns:
            Dict[int, bool]: Confirmed violations per slot
        """
        if timestamp is None:
            timestamp = datetime.now()

        confirmed_violations = {}

        for slot_id, (is_improper, alignment_score, severity_score, confidence) in slot_observations.items():
            # Initialize slot state if new
            if slot_id not in self.slot_states:
                self.slot_states[slot_id] = SlotTemporalState(slot_id=slot_id)

            # Add observation
            observation = FrameObservation(
                timestamp=timestamp,
                is_improperly_parked=is_improper,
                alignment_score=alignment_score,
                severity_score=severity_score,
                confidence=confidence
            )
            self.slot_states[slot_id].add_observation(observation)

            # Prune old observations (keep sliding window)
            if len(self.slot_states[slot_id].observations) > self.window_size:
                self.slot_states[slot_id].observations.pop(0)

            # Check if violation is confirmed
            is_confirmed = self._check_violation_confirmed(slot_id)
            confirmed_violations[slot_id] = is_confirmed
            self.slot_states[slot_id].violation_confirmed = is_confirmed

        return confirmed_violations

    def _check_violation_confirmed(self, slot_id: int) -> bool:
        """
        Check if violation is confirmed based on temporal threshold.
        
        ∑(V_t) ≥ T_threshold
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            bool: True if violation confirmed, False otherwise
        """
        state = self.slot_states[slot_id]
        violation_count = sum(1 for obs in state.observations if obs.is_improperly_parked)
        return violation_count >= self.confirmation_threshold

    def get_violation_severity(self, slot_id: int) -> float:
        """
        Get average severity score for confirmed violation.
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            float: Average severity score [0, 1]
        """
        if slot_id not in self.slot_states:
            return 0.0

        state = self.slot_states[slot_id]
        violations = [obs for obs in state.observations if obs.is_improperly_parked]

        if not violations:
            return 0.0

        return float(np.mean([v.severity_score for v in violations]))

    def get_violation_duration_frames(self, slot_id: int) -> int:
        """
        Get number of consecutive violation frames.
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            int: Number of frames with violations
        """
        if slot_id not in self.slot_states:
            return 0

        state = self.slot_states[slot_id]
        return sum(1 for obs in state.observations if obs.is_improperly_parked)

    def get_violation_duration_seconds(self, slot_id: int) -> float:
        """
        Get total violation duration in seconds.
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            float: Duration in seconds
        """
        if slot_id not in self.slot_states:
            return 0.0

        return self.slot_states[slot_id].get_violation_duration()

    def reset_slot(self, slot_id: int) -> None:
        """
        Reset temporal tracking for a slot (e.g., vehicle left).
        
        Args:
            slot_id: Parking slot ID
        """
        if slot_id in self.slot_states:
            self.slot_states[slot_id] = SlotTemporalState(slot_id=slot_id)

    def get_summary(self, slot_id: int) -> Dict:
        """
        Get comprehensive temporal summary for a slot.
        
        Args:
            slot_id: Parking slot ID
            
        Returns:
            Dict: Temporal summary with statistics
        """
        if slot_id not in self.slot_states:
            return {}

        state = self.slot_states[slot_id]
        observations = state.observations

        if not observations:
            return {
                "slot_id": slot_id,
                "num_observations": 0,
                "violation_confirmed": False,
                "violation_count": 0
            }

        violation_obs = [obs for obs in observations if obs.is_improperly_parked]
        proper_obs = [obs for obs in observations if not obs.is_improperly_parked]

        return {
            "slot_id": slot_id,
            "num_observations": len(observations),
            "violation_confirmed": state.violation_confirmed,
            "violation_count": state.violation_count,
            "violation_ratio": len(violation_obs) / len(observations) if observations else 0,
            "avg_alignment_score": np.mean([o.alignment_score for o in observations]),
            "avg_violation_severity": np.mean([o.severity_score for o in violation_obs]) if violation_obs else 0,
            "avg_confidence": np.mean([o.confidence for o in observations]),
            "duration_seconds": state.get_violation_duration(),
            "first_violation_time": state.first_violation_time,
            "last_violation_time": state.last_violation_time
        }

    def get_all_confirmed_violations(self) -> List[int]:
        """
        Get list of all confirmed violated slots.
        
        Returns:
            List[int]: Slot IDs with confirmed violations
        """
        return [slot_id for slot_id, state in self.slot_states.items()
                if state.violation_confirmed]

    def clear_history(self) -> None:
        """Clear all temporal tracking data."""
        self.slot_states.clear()
