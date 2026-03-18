"""
Dynamic Allocation Strategy

Advanced allocation combining:
- Vision information (alignment score)
- Temporal verification (confirmed violations)
- Ranking logic
- Adaptive thresholds

This is the proposed method that uses all components.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class DynamicAllocationDecision:
    """Decision from dynamic allocator."""
    vehicle_id: int
    slot_id: Optional[int]
    decision_score: float
    reason: str
    uses_temporal: bool
    uses_alignment: bool


class DynamicAllocationModule:
    """
    Advanced allocation with temporal + spatial reasoning.
    
    Key innovation: Uses temporal verification to confirm violations
    before making allocation decisions.
    """

    def __init__(self,
                 spatial_weight: float = 0.5,
                 temporal_weight: float = 0.3,
                 confidence_weight: float = 0.2,
                 violation_penalty: float = 0.7):
        """
        Initialize dynamic allocator.
        
        Args:
            spatial_weight: Weight for alignment score
            temporal_weight: Weight for temporal confirmation
            confidence_weight: Weight for detector confidence
            violation_penalty: Penalty multiplier for confirmed violations
        """
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.confidence_weight = confidence_weight
        self.violation_penalty = violation_penalty

        # Normalize weights
        total = spatial_weight + temporal_weight + confidence_weight
        self.spatial_weight /= total
        self.temporal_weight /= total
        self.confidence_weight /= total

        self.allocations: Dict[int, DynamicAllocationDecision] = {}

    def decide_allocation(self,
                         vehicle_id: int,
                         detector_confidence: float,
                         slot_candidates: Dict[int, Dict]) -> DynamicAllocationDecision:
        """
        Make dynamic allocation decision for a vehicle.
        
        Args:
            vehicle_id: Vehicle ID
            detector_confidence: YOLO confidence
            slot_candidates: Dict mapping slot_id to {
                'alignment_score': float,
                'iou': float,
                'is_improper': bool,
                'temporal_confirmed': bool,
                'temporal_severity': float
            }
            
        Returns:
            DynamicAllocationDecision: Allocation decision with reasoning
        """
        if not slot_candidates:
            return DynamicAllocationDecision(
                vehicle_id=vehicle_id,
                slot_id=None,
                decision_score=0.0,
                reason="No available slots",
                uses_temporal=False,
                uses_alignment=False
            )

        best_slot = None
        best_score = 0.0
        best_reason = ""

        for slot_id, metrics in slot_candidates.items():
            alignment = metrics.get('alignment_score', 0.0)
            iou = metrics.get('iou', 0.0)
            is_improper = metrics.get('is_improper', False)
            temporal_confirmed = metrics.get('temporal_confirmed', False)
            temporal_severity = metrics.get('temporal_severity', 0.0)

            # Compute decision score
            # Base components
            spatial_component = self.spatial_weight * alignment
            confidence_component = self.confidence_weight * detector_confidence

            # Temporal component
            if temporal_confirmed:
                temporal_component = self.temporal_weight * (1.0 - temporal_severity)
                uses_temporal = True
            else:
                temporal_component = self.temporal_weight * 0.5  # Uncertainty
                uses_temporal = False

            # Raw score
            score = spatial_component + temporal_component + confidence_component

            # Apply violation penalty
            if temporal_confirmed and is_improper:
                score *= self.violation_penalty
                reason = f"Temporal violation penalty applied (severity: {temporal_severity:.2f})"
            elif is_improper and not temporal_confirmed:
                score *= 0.85
                reason = "Spatial violation detected, awaiting temporal confirmation"
            else:
                reason = f"Valid allocation candidate"

            if score > best_score and score >= 0.3:
                best_score = score
                best_slot = slot_id
                best_reason = reason

        if best_slot is None:
            return DynamicAllocationDecision(
                vehicle_id=vehicle_id,
                slot_id=None,
                decision_score=0.0,
                reason="No suitable slot found (low scores)",
                uses_temporal=False,
                uses_alignment=False
            )

        decision = DynamicAllocationDecision(
            vehicle_id=vehicle_id,
            slot_id=best_slot,
            decision_score=best_score,
            reason=best_reason,
            uses_temporal=True,
            uses_alignment=True
        )

        self.allocations[vehicle_id] = decision
        return decision

    def batch_allocate(self,
                      vehicle_data: List[Dict]) -> List[DynamicAllocationDecision]:
        """
        Allocate multiple vehicles.
        
        Args:
            vehicle_data: List of {
                'vehicle_id': int,
                'detector_confidence': float,
                'slot_candidates': Dict
            }
            
        Returns:
            List[DynamicAllocationDecision]: Decisions for each vehicle
        """
        decisions = []
        for data in vehicle_data:
            decision = self.decide_allocation(
                data['vehicle_id'],
                data['detector_confidence'],
                data['slot_candidates']
            )
            decisions.append(decision)

        return decisions

    def get_allocation(self, vehicle_id: int) -> Optional[DynamicAllocationDecision]:
        """Get allocation decision for vehicle."""
        return self.allocations.get(vehicle_id)

    def get_summary(self) -> Dict:
        """Get summary of allocations."""
        if not self.allocations:
            return {"total": 0, "successful": 0, "failed": 0}

        successful = sum(1 for d in self.allocations.values() if d.slot_id is not None)
        failed = len(self.allocations) - successful

        return {
            "total": len(self.allocations),
            "successful": successful,
            "failed": failed,
            "avg_score": float(np.mean([d.decision_score for d in self.allocations.values()]))
        }

    def clear(self) -> None:
        """Clear allocations."""
        self.allocations.clear()
