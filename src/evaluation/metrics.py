"""
Evaluation Metrics

Comprehensive evaluation metrics for parking violation detection system.

Metrics:
- Precision, Recall, F1 for detection
- Alignment accuracy
- Temporal verification accuracy
- Penalty scoring correlation
"""

from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass
class AlignmentMetrics:
    """Alignment accuracy metrics."""
    mae_alignment: float  # Mean absolute error
    rmse_alignment: float
    correlation_iou_alignment: float
    proper_parking_accuracy: float


@dataclass
class TemporalMetrics:
    """Temporal verification metrics."""
    confirmation_accuracy: float
    early_detection_rate: float
    false_positive_reduction: float  # % reduction vs spatial-only
    avg_confirmation_frames: float


class MetricsCalculator:
    """Calculate evaluation metrics."""

    @staticmethod
    def compute_detection_metrics(predictions: List[bool],
                                 ground_truth: List[bool]) -> DetectionMetrics:
        """
        Compute detection performance metrics.
        
        Args:
            predictions: Predicted labels [improper=1, proper=0]
            ground_truth: Ground truth labels
            
        Returns:
            DetectionMetrics: Comprehensive detection metrics
        """
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)

        tp = np.sum((predictions == 1) & (ground_truth == 1))
        fp = np.sum((predictions == 1) & (ground_truth == 0))
        fn = np.sum((predictions == 0) & (ground_truth == 1))
        tn = np.sum((predictions == 0) & (ground_truth == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else 0.0

        return DetectionMetrics(
            precision=float(precision),
            recall=float(recall),
            f1_score=float(f1),
            accuracy=float(accuracy),
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn),
            true_negatives=int(tn)
        )

    @staticmethod
    def compute_alignment_metrics(predicted_scores: List[float],
                                 ground_truth_scores: List[float]) -> AlignmentMetrics:
        """
        Compute alignment accuracy metrics.
        
        Args:
            predicted_scores: Predicted alignment scores
            ground_truth_scores: Ground truth alignment scores
            
        Returns:
            AlignmentMetrics: Alignment accuracy measures
        """
        predicted = np.array(predicted_scores)
        ground_truth = np.array(ground_truth_scores)

        mae = np.mean(np.abs(predicted - ground_truth))
        rmse = np.sqrt(np.mean((predicted - ground_truth) ** 2))

        correlation = np.corrcoef(predicted, ground_truth)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        # Proper parking accuracy (alignment > 0.5)
        pred_proper = predicted >= 0.5
        gt_proper = ground_truth >= 0.5
        proper_accuracy = np.mean(pred_proper == gt_proper)

        return AlignmentMetrics(
            mae_alignment=float(mae),
            rmse_alignment=float(rmse),
            correlation_iou_alignment=float(correlation),
            proper_parking_accuracy=float(proper_accuracy)
        )

    @staticmethod
    def compute_temporal_metrics(spatial_predictions: List[bool],
                                temporal_predictions: List[bool],
                                ground_truth: List[bool]) -> TemporalMetrics:
        """
        Compute temporal verification metrics.
        
        Args:
            spatial_predictions: Spatial-only predictions (alignment)
            temporal_predictions: Temporal-confirmed predictions
            ground_truth: Ground truth violations
            
        Returns:
            TemporalMetrics: Temporal verification performance
        """
        spatial_predictions = np.array(spatial_predictions)
        temporal_predictions = np.array(temporal_predictions)
        ground_truth = np.array(ground_truth)

        # Confirmation accuracy: % of temporal that match ground truth
        temporal_accuracy = np.mean(temporal_predictions == ground_truth)

        # Early detection: detections before temporal confirmation
        # (simplified: count cases where spatial detected, temporal eventually confirmed)
        early_detections = np.sum((spatial_predictions == 1) & (ground_truth == 1))
        confirmed_detections = np.sum((temporal_predictions == 1) & (ground_truth == 1))
        early_rate = confirmed_detections / (early_detections + 1e-6)

        # False positive reduction
        spatial_fp = np.sum((spatial_predictions == 1) & (ground_truth == 0))
        temporal_fp = np.sum((temporal_predictions == 1) & (ground_truth == 0))
        fp_reduction = (spatial_fp - temporal_fp) / (spatial_fp + 1e-6)

        return TemporalMetrics(
            confirmation_accuracy=float(temporal_accuracy),
            early_detection_rate=float(early_rate),
            false_positive_reduction=float(max(0, fp_reduction)),  # Clip to [0, ∞)
            avg_confirmation_frames=3.0  # Depends on implementation
        )

    @staticmethod
    def compute_penalty_correlation(predicted_penalties: List[float],
                                   ground_truth_violations: List[Tuple[float, float, int]]) -> Dict:
        """
        Compute penalty scoring correlation with ground truth.
        
        Args:
            predicted_penalties: Predicted penalty scores
            ground_truth_violations: List of (severity, duration, count) tuples
            
        Returns:
            Dict: Correlation metrics
        """
        predicted = np.array(predicted_penalties)

        # Compute expected penalties from ground truth
        expected = []
        for severity, duration, count in ground_truth_violations:
            # Simple expected: 0.3*duration/3600 + 0.4*severity + 0.3*(count/5)
            exp = 0.3 * (min(duration, 3600) / 3600) + 0.4 * severity + 0.3 * (min(count, 5) / 5)
            expected.append(exp)

        expected = np.array(expected)

        # Compute correlation
        correlation = np.corrcoef(predicted, expected)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        mae = np.mean(np.abs(predicted - expected))
        rmse = np.sqrt(np.mean((predicted - expected) ** 2))

        return {
            "correlation": float(correlation),
            "mae": float(mae),
            "rmse": float(rmse)
        }


class ExperimentRunner:
    """Run complete evaluation experiments."""

    def __init__(self, name: str):
        """Initialize experiment runner."""
        self.name = name
        self.results = {}

    def evaluate_detection_module(self,
                                 predictions: List[bool],
                                 ground_truth: List[bool],
                                 description: str = "") -> DetectionMetrics:
        """
        Evaluate detection module.
        
        Args:
            predictions: Predicted improper parking labels
            ground_truth: Ground truth labels
            description: Experiment description
            
        Returns:
            DetectionMetrics: Evaluation results
        """
        metrics = MetricsCalculator.compute_detection_metrics(predictions, ground_truth)
        self.results[f"detection_{description}"] = metrics
        return metrics

    def evaluate_alignment_module(self,
                                 predicted_scores: List[float],
                                 ground_truth_scores: List[float],
                                 description: str = "") -> AlignmentMetrics:
        """Evaluate alignment scoring accuracy."""
        metrics = MetricsCalculator.compute_alignment_metrics(predicted_scores, ground_truth_scores)
        self.results[f"alignment_{description}"] = metrics
        return metrics

    def evaluate_temporal_module(self,
                                spatial_pred: List[bool],
                                temporal_pred: List[bool],
                                ground_truth: List[bool],
                                description: str = "") -> TemporalMetrics:
        """Evaluate temporal verification effectiveness."""
        metrics = MetricsCalculator.compute_temporal_metrics(spatial_pred, temporal_pred, ground_truth)
        self.results[f"temporal_{description}"] = metrics
        return metrics

    def generate_summary(self) -> Dict:
        """Generate experiment summary."""
        return {
            "experiment_name": self.name,
            "num_evaluations": len(self.results),
            "results": {k: v.__dict__ for k, v in self.results.items()}
        }

    def print_summary(self) -> None:
        """Print human-readable summary."""
        print(f"\n{'='*60}")
        print(f"Experiment: {self.name}")
        print(f"{'='*60}")

        for name, metrics in self.results.items():
            print(f"\n{name}:")
            if hasattr(metrics, '__dict__'):
                for key, value in metrics.__dict__.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
