"""
Experiments

Complete end-to-end experimental framework for evaluating the smart parking system.
Runs through detection → alignment → temporal → allocation → penalization pipeline.
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta

from src.detection.vehicle_detector import YOLOVehicleDetector, Detection
from src.detection.slot_alignment import SlotAlignmentModule, AlignmentResult
from src.detection.occupancy_classifier import OccupancyClassifier, OccupancyStatus
from src.temporal.temporal_verification import TemporalVerificationModule, FrameObservation
from src.allocation.static_allocation import StaticAllocationModule
from src.allocation.sorting_allocation import SortingAllocationModule
from src.allocation.dynamic_allocation import DynamicAllocationModule
from src.penalization.penalty_engine import PenaltyEngine
from src.evaluation.metrics import MetricsCalculator, ExperimentRunner
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SmartParkingExperiment:
    """
    Complete smart parking system experiment.
    
    Tests the full pipeline: Detection → Alignment → Temporal → Allocation → Penalization
    """

    def __init__(self, experiment_name: str, results_dir: str = "results"):
        """
        Initialize experiment.
        
        Args:
            experiment_name: Name of experiment
            results_dir: Directory to save results
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Initialize modules
        self.detector = None  # YOLO detector (optional, requires model)
        self.alignment = SlotAlignmentModule(delta=0.5, epsilon=0.3)
        self.occupancy = OccupancyClassifier()
        self.temporal = TemporalVerificationModule(confirmation_threshold=3, window_size=10)
        self.penalty_engine = PenaltyEngine(alpha=0.3, beta=0.4, gamma=0.3)

        # Allocation strategies
        self.static_allocator = StaticAllocationModule()
        self.sorting_allocator = SortingAllocationModule()
        self.dynamic_allocator = DynamicAllocationModule(
            spatial_weight=0.5, temporal_weight=0.3, confidence_weight=0.2
        )

        # Metrics runner
        self.metrics_runner = ExperimentRunner(experiment_name)

        self.experiment_log = []

    def run_frame_processing(self,
                           frame_id: int,
                           detections: List[Detection],
                           slot_polygons: Dict[int, 'Polygon'],
                           ground_truth_violations: Dict[int, bool] = None,
                           timestamp: datetime = None) -> Dict:
        """
        Process a single frame through the full pipeline.
        
        Args:
            frame_id: Frame number
            detections: List of detected vehicles
            slot_polygons: Dict mapping slot_id to Polygon
            ground_truth_violations: Optional ground truth violations per slot
            timestamp: Frame timestamp
            
        Returns:
            Dict: Complete frame processing results
        """
        if timestamp is None:
            timestamp = datetime.now()

        frame_result = {
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "detections": len(detections),
            "alignment_results": {},
            "temporal_confirmations": {},
            "occupancy_status": {},
            "penalties": {}
        }

        # Step 1: Alignment analysis
        alignment_results = {}
        slot_confidences = {}
        slot_alignment_scores = {}

        for slot_id, slot_polygon in slot_polygons.items():
            # Find detection for this slot (simplified: match closest bbox)
            matching_detection = self._find_detection_for_slot(detections, slot_polygon)

            if matching_detection:
                result = self.alignment.analyze_vehicle_slot_alignment(
                    matching_detection.bbox,
                    slot_polygon,
                    matching_detection.confidence
                )
                alignment_results[slot_id] = result

                # Compute slot confidence
                slot_conf = self.alignment.compute_slot_confidence(result, matching_detection.confidence)
                slot_confidences[slot_id] = slot_conf
                slot_alignment_scores[slot_id] = result.alignment_score

                frame_result["alignment_results"][slot_id] = {
                    "iou": result.iou,
                    "alignment_score": result.alignment_score,
                    "is_improper": result.is_improperly_parked,
                    "severity": result.severity_score
                }
            else:
                slot_confidences[slot_id] = 0.0
                slot_alignment_scores[slot_id] = 0.0

        # Step 2: Temporal verification
        slot_observations = {}
        for slot_id, result in alignment_results.items():
            slot_observations[slot_id] = (
                result.is_improperly_parked,
                result.alignment_score,
                result.severity_score,
                slot_confidences[slot_id]
            )

        confirmed_violations = self.temporal.process_frame_observations(
            frame_id, slot_observations, timestamp
        )

        frame_result["temporal_confirmations"] = confirmed_violations

        # Step 3: Occupancy classification
        for slot_id in slot_polygons.keys():
            has_detection = slot_id in alignment_results
            status = self.occupancy.classify(
                has_detection,
                detection_confidence=slot_confidences.get(slot_id, 0),
                alignment_score=alignment_results.get(slot_id, AlignmentResult(
                    iou=0, center_deviation=1, alignment_score=0,
                    is_improperly_parked=False, severity_score=0,
                    overlap_percentage=0
                )).alignment_score,
                iou=alignment_results.get(slot_id, AlignmentResult(
                    iou=0, center_deviation=1, alignment_score=0,
                    is_improperly_parked=False, severity_score=0,
                    overlap_percentage=0
                )).iou,
                temporal_confirmed=confirmed_violations.get(slot_id, False)
            )
            frame_result["occupancy_status"][slot_id] = status.value

        # Step 4: Dynamic allocation
        for slot_id, result in alignment_results.items():
            if confirmed_violations.get(slot_id, False):
                # Compute penalty for this violation
                severity = self.temporal.get_violation_severity(slot_id)
                duration = self.temporal.get_violation_duration_seconds(slot_id)
                violation_count = self.temporal.get_violation_duration_frames(slot_id)

                penalty_breakdown = self.penalty_engine.compute_penalty(
                    slot_id, severity, duration, violation_count
                )
                frame_result["penalties"][slot_id] = {
                    "total_penalty": penalty_breakdown.total_penalty,
                    "duration_component": penalty_breakdown.duration_component,
                    "severity_component": penalty_breakdown.severity_component,
                    "repetition_component": penalty_breakdown.repetition_component
                }

        self.experiment_log.append(frame_result)
        return frame_result

    def run_batch_experiment(self,
                            frame_data: List[Dict],
                            slot_polygons: Dict[int, 'Polygon']) -> Dict:
        """
        Run complete batch experiment across multiple frames.
        
        Args:
            frame_data: List of dicts with frame_id, detections, ground_truth_violations
            slot_polygons: Slot polygon definitions
            
        Returns:
            Dict: Complete experiment results
        """
        logger.info("="*60)
        logger.info("Running Experiment: %s", self.experiment_name)
        logger.info("Processing %d frames ...", len(frame_data))

        for i, data in enumerate(frame_data):
            result = self.run_frame_processing(
                data['frame_id'],
                data['detections'],
                slot_polygons,
                data.get('ground_truth_violations'),
                data.get('timestamp')
            )

            if (i + 1) % max(1, len(frame_data) // 10) == 0:
                logger.info("  Progress: %d/%d", i + 1, len(frame_data))

        logger.info("Experiment complete")
        return self.generate_results()

    def _find_detection_for_slot(self, detections: List[Detection], slot_polygon) -> Detection:
        """Find detection with highest overlap for slot."""
        best_detection = None
        best_overlap = 0

        for det in detections:
            overlap = self.alignment.geometry.bbox_overlap_percentage(det.bbox, slot_polygon)
            if overlap > best_overlap:
                best_overlap = overlap
                best_detection = det

        return best_detection if best_overlap > 0.1 else None

    def generate_results(self) -> Dict:
        """Generate complete experiment results."""
        total_frames = len(self.experiment_log)
        total_violations = sum(
            len(frame['penalties']) for frame in self.experiment_log
        )

        return {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_frames": total_frames,
            "total_violations_detected": total_violations,
            "avg_violations_per_frame": total_violations / max(1, total_frames),
            "log_file": str(self.results_dir / f"{self.experiment_name}_log.json")
        }

    def save_results(self) -> None:
        """Save experiment results to file."""
        results = self.generate_results()
        log_path = self.results_dir / f"{self.experiment_name}_log.json"

        with open(log_path, 'w') as f:
            json.dump(self.experiment_log, f, indent=2, default=str)

        logger.info("Results saved to %s", log_path)

    def print_summary(self) -> None:
        """Log experiment summary."""
        results = self.generate_results()
        logger.info("="*60)
        logger.info("Experiment Summary: %s", results["experiment_name"])
        logger.info("  Total frames processed:   %d", results["total_frames"])
        logger.info("  Total violations detected: %d", results["total_violations_detected"])
        logger.info("  Avg violations/frame:      %.2f", results["avg_violations_per_frame"])
        logger.info("  Results saved: %s", results["log_file"])
