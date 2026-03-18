"""
Smart Parking System - Main Entry Point

Pipeline:
    YOLO Detection → Slot Alignment → Temporal Verification
                   → Dynamic Allocation → Penalty Engine

All output (stdout equivalent) and errors are routed through Python's
logging module so that HPC job schedulers (SLURM, PBS, etc.) capture
everything in their standard log files *and* in our own timestamped
logs/ directory.
"""

# ── stdlib ─────────────────────────────────────────────────────────────────
import json
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── third-party ─────────────────────────────────────────────────────────────
from shapely.geometry import Polygon

# ── logging must be configured before any project import that calls
#    get_logger() at module level ───────────────────────────────────────────
from src.utils.logger import setup_logging, get_logger

_log_file = setup_logging()          # writes to logs/smart_parking_YYYYMMDD_HHMMSS.log
logger = get_logger(__name__)

# ── project imports ─────────────────────────────────────────────────────────
from src.detection.vehicle_detector import YOLOVehicleDetector, Detection
from src.detection.slot_alignment import SlotAlignmentModule
from src.detection.occupancy_classifier import OccupancyClassifier
from src.temporal.temporal_verification import TemporalVerificationModule
from src.allocation.dynamic_allocation import DynamicAllocationModule
from src.allocation.static_allocation import StaticAllocationModule
from src.allocation.sorting_allocation import SortingAllocationModule
from src.penalization.penalty_engine import PenaltyEngine
from src.evaluation.experiments import SmartParkingExperiment


# ═══════════════════════════════════════════════════════════════════════════
class SmartParkingSystem:
    """Main orchestrator – wires all pipeline stages together."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        logger.info("Configuration loaded: %s", self.config)

        self.slot_polygons = self._load_slot_coordinates()
        logger.info("Loaded %d parking slot polygons", len(self.slot_polygons))

        self.detector = YOLOVehicleDetector(model_path="yolov8n.pt")

        self.alignment = SlotAlignmentModule(
            delta=self.config.get("delta", 0.5),
            epsilon=self.config.get("epsilon", 0.3),
        )
        self.occupancy = OccupancyClassifier(
            alignment_threshold=self.config.get("alignment_threshold", 0.5),
            iou_threshold=self.config.get("iou_threshold", 0.3),
        )
        self.temporal = TemporalVerificationModule(
            confirmation_threshold=self.config.get("confirmation_threshold", 3),
            window_size=self.config.get("window_size", 10),
        )
        self.penalty_engine = PenaltyEngine(
            alpha=self.config.get("alpha", 0.3),
            beta=self.config.get("beta", 0.4),
            gamma=self.config.get("gamma", 0.3),
        )
        self.static_allocator  = StaticAllocationModule()
        self.sorting_allocator = SortingAllocationModule()
        self.dynamic_allocator = DynamicAllocationModule()

        logger.info(
            "SmartParkingSystem ready  |  detector=%s  slots=%d",
            "loaded" if self.detector.is_loaded else "not loaded",
            len(self.slot_polygons),
        )

    # ── config ──────────────────────────────────────────────────────────────

    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path:
            p = Path(config_path)
            if p.exists():
                with open(p) as f:
                    cfg = json.load(f)
                logger.debug("Config read from %s", p)
                return cfg
            logger.warning("Config file not found: %s  – using defaults", config_path)
        return {
            "delta": 0.5,
            "epsilon": 0.3,
            "alignment_threshold": 0.5,
            "iou_threshold": 0.3,
            "confirmation_threshold": 3,
            "window_size": 10,
            "alpha": 0.3,
            "beta": 0.4,
            "gamma": 0.3,
        }

    def _load_slot_coordinates(self) -> Dict[int, Polygon]:
        slot_path = Path("data/cnrpark/slots_coordinates.json")
        if not slot_path.exists():
            logger.warning("slots_coordinates.json not found – using 3 synthetic slots")
            return {
                1: Polygon([(0, 0), (100, 0), (100, 50), (0, 50)]),
                2: Polygon([(110, 0), (210, 0), (210, 50), (110, 50)]),
                3: Polygon([(220, 0), (320, 0), (320, 50), (220, 50)]),
            }

        with open(slot_path) as f:
            slot_data = json.load(f)

        slot_polygons: Dict[int, Polygon] = {}
        for slot_id, slot_info in slot_data.items():
            coords = (
                slot_info.get("coordinates", [])
                if isinstance(slot_info, dict)
                else slot_info
            )
            if not coords or not isinstance(coords[0], (list, tuple)):
                continue
            try:
                slot_polygons[int(slot_id)] = Polygon(coords)
            except (ValueError, TypeError) as exc:
                logger.debug("Skipping slot %s – bad coords: %s", slot_id, exc)

        if not slot_polygons:
            logger.warning("No valid polygons in slots_coordinates.json – generating synthetic")
            for i in range(1, 6):
                slot_polygons[i] = Polygon(
                    [[i * 200, i * 200], [i * 200 + 150, i * 200],
                     [i * 200 + 150, i * 200 + 150], [i * 200, i * 200 + 150]]
                )

        return slot_polygons

    # ── core processing ──────────────────────────────────────────────────────

    def process_frame(self, image_path: str, frame_id: int,
                      timestamp: Optional[datetime] = None) -> Dict:
        if timestamp is None:
            timestamp = datetime.now()

        logger.debug("Processing frame %d  |  image=%s", frame_id, image_path)

        # 1 – Detection
        detections: List[Detection] = []
        if self.detector:
            detections = self.detector.detect(image_path)
        logger.debug("Frame %d  |  %d detections", frame_id, len(detections))

        # 2 – Alignment
        alignment_results: Dict = {}
        slot_confidences: Dict = {}
        for slot_id, slot_polygon in self.slot_polygons.items():
            matching = self._find_detection_for_slot(detections, slot_polygon)
            if matching:
                result = self.alignment.analyze_vehicle_slot_alignment(
                    matching.bbox, slot_polygon, matching.confidence
                )
                alignment_results[slot_id] = result
                slot_confidences[slot_id] = self.alignment.compute_slot_confidence(
                    result, matching.confidence
                )

        # 3 – Temporal verification
        slot_obs = {
            sid: (r.is_improperly_parked, r.alignment_score,
                  r.severity_score, slot_confidences[sid])
            for sid, r in alignment_results.items()
        }
        confirmed_violations = self.temporal.process_frame_observations(
            frame_id, slot_obs, timestamp
        )

        # 4 – Occupancy classification
        occupancy_status: Dict = {}
        for slot_id in self.slot_polygons:
            has_det = slot_id in alignment_results
            status = self.occupancy.classify(
                has_det,
                detection_confidence=slot_confidences.get(slot_id, 0),
                alignment_score=alignment_results.get(slot_id, None),
                temporal_confirmed=confirmed_violations.get(slot_id, False),
            )
            occupancy_status[slot_id] = status.value

        # 5 – Penalties
        penalties: Dict = {}
        for slot_id, is_confirmed in confirmed_violations.items():
            if is_confirmed:
                severity = self.temporal.get_violation_severity(slot_id)
                duration = self.temporal.get_violation_duration_seconds(slot_id)
                vcount   = self.temporal.get_violation_duration_frames(slot_id)
                pb = self.penalty_engine.compute_penalty(
                    slot_id, severity, duration, vcount
                )
                penalties[slot_id] = pb.total_penalty
                logger.debug(
                    "Frame %d  |  slot %d VIOLATION  penalty=%.2f",
                    frame_id, slot_id, pb.total_penalty,
                )

        return {
            "frame_id":             frame_id,
            "timestamp":            timestamp.isoformat(),
            "detections":           len(detections),
            "alignment_results":    alignment_results,
            "confirmed_violations": confirmed_violations,
            "occupancy_status":     occupancy_status,
            "penalties":            penalties,
        }

    def _find_detection_for_slot(self, detections: List[Detection],
                                  slot_polygon: Polygon):
        best, best_overlap = None, 0.0
        for det in detections:
            ov = self.alignment.geometry.bbox_overlap_percentage(det.bbox, slot_polygon)
            if ov > best_overlap:
                best_overlap, best = ov, det
        return best if best_overlap > 0.1 else None

    def generate_report(self, output_path: Optional[str] = None) -> Dict:
        report = {
            "system_name": "Smart Parking System",
            "timestamp":   datetime.now().isoformat(),
            "configuration": self.config,
            "components": {
                "detection":    "YOLO (YOLOv8)",
                "alignment":    "Novelty: IoU x (1-D) scoring",
                "temporal":     "Frame-based confirmation",
                "allocation":   "Static/Sorting/Dynamic strategies",
                "penalization": "alpha*Duration + beta*Severity + gamma*Repetition",
            },
            "slot_count": len(self.slot_polygons),
            "status": "ready",
        }
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("System report saved to %s", output_path)
        return report


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset processing helpers
# ═══════════════════════════════════════════════════════════════════════════

def process_dataset(system: SmartParkingSystem, image_dir: Path) -> List[Dict]:
    all_results: List[Dict] = []
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    images_set: set = set()
    for ext in exts:
        images_set.update(image_dir.glob(ext))
        images_set.update(image_dir.glob(ext.upper()))
    images = sorted(images_set, key=lambda x: x.name.lower())

    if not images:
        logger.warning("No images found in %s", image_dir)
        return all_results

    logger.info("Found %d images in %s", len(images), image_dir)

    for frame_id, image_path in enumerate(images):
        try:
            result = system.process_frame(str(image_path), frame_id)
            serializable = {
                "frame_id":   result["frame_id"],
                "timestamp":  result["timestamp"],
                "image_path": image_path.name,
                "detections": result["detections"],
                "confirmed_violations": result["confirmed_violations"],
                "occupancy_status":     result["occupancy_status"],
                "penalties":            result["penalties"],
                "alignment_summary": {
                    sid: {
                        "alignment_score": ar.alignment_score,
                        "is_improper":     ar.is_improperly_parked,
                        "severity":        ar.severity_score,
                        "iou":             ar.iou,
                    }
                    for sid, ar in result["alignment_results"].items()
                },
            }
            all_results.append(serializable)

            if (frame_id + 1) % 10 == 0 or frame_id == len(images) - 1:
                violations = sum(1 for v in result["confirmed_violations"].values() if v)
                logger.info(
                    "Progress: %d/%d frames  |  violations this frame: %d",
                    frame_id + 1, len(images), violations,
                )
        except Exception:
            logger.error("Error processing %s\n%s", image_path.name, traceback.format_exc())

    return all_results


def compute_summary_statistics(results: List[Dict]) -> Dict:
    if not results:
        logger.warning("No results to summarise")
        return {"error": "No results to summarize"}

    total_violations = 0
    total_penalties  = 0.0
    slot_violation_counts: Dict = {}
    occupancy_counts = {"occupied": 0, "empty": 0, "improper": 0, "unknown": 0}

    for result in results:
        for sid, is_v in result["confirmed_violations"].items():
            if is_v:
                total_violations += 1
                slot_violation_counts[sid] = slot_violation_counts.get(sid, 0) + 1
        total_penalties += sum(result["penalties"].values())
        for sid, status in result["occupancy_status"].items():
            occupancy_counts[status] = occupancy_counts.get(status, 0) + 1

    top_slots = sorted(slot_violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "frames_processed":              len(results),
        "total_violations_detected":     total_violations,
        "total_penalty_score":           round(total_penalties, 2),
        "average_penalty_per_violation": round(total_penalties / max(total_violations, 1), 2),
        "occupancy_distribution":        occupancy_counts,
        "unique_slots_with_violations":  len(slot_violation_counts),
        "top_violating_slots": [
            {"slot_id": sid, "violation_count": cnt} for sid, cnt in top_slots
        ],
        "violation_rate": round(total_violations / max(len(results), 1), 2),
    }


def save_results(results: List[Dict], summary: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "processing_results.json"
    with open(results_path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                   "summary": summary, "frame_results": results}, f, indent=2)
    logger.info("Detailed results  -> %s", results_path)

    summary_path = output_dir / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), **summary}, f, indent=2)
    logger.info("Summary report    -> %s", summary_path)

    violations_path = output_dir / "violations.csv"
    with open(violations_path, "w") as f:
        f.write("frame_id,image,slot_id,penalty_score\n")
        for r in results:
            for sid, penalty in r["penalties"].items():
                f.write(f"{r['frame_id']},{r['image_path']},{sid},{penalty:.2f}\n")
    logger.info("Violations CSV    -> %s", violations_path)


# ── CNR-EXT-Patches pipeline ────────────────────────────────────────────────

def load_cnr_ext_patches(dataset_path: Path,
                          max_samples: Optional[int] = None) -> Tuple[List[Dict], Dict]:
    labels_file = dataset_path / "LABELS" / "all.txt"
    patches_dir = dataset_path / "PATCHES"

    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    logger.info("Reading labels from %s", labels_file)
    with open(labels_file) as f:
        lines = f.readlines()

    total_available = len(lines)
    if max_samples and max_samples < total_available:
        random.shuffle(lines)
        lines = lines[:max_samples]
        logger.info("Sampling %d of %d available labels", max_samples, total_available)

    samples: List[Dict] = []
    weather_counts: Dict[str, int] = {}
    occupancy_counts: Dict[str, int] = {"empty": 0, "occupied": 0}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        rel_path, label_str = parts
        label = int(label_str)
        image_path = patches_dir / rel_path
        if not image_path.exists():
            continue
        weather = rel_path.split("/")[0]
        weather_counts[weather] = weather_counts.get(weather, 0) + 1
        fname  = image_path.name
        fparts = fname.replace(".jpg", "").split("_")
        camera  = fparts[3] if len(fparts) >= 5 else "unknown"
        slot_id = int(fparts[4]) if len(fparts) >= 5 else 0
        occupancy = "occupied" if label == 1 else "empty"
        occupancy_counts[occupancy] += 1
        samples.append({
            "image_path": str(image_path),
            "label":      label,
            "occupancy":  occupancy,
            "weather":    weather,
            "camera":     camera,
            "slot_id":    slot_id,
            "filename":   fname,
        })

    metadata = {
        "total_samples":          len(samples),
        "weather_distribution":   weather_counts,
        "occupancy_distribution": occupancy_counts,
        "dataset_path":           str(dataset_path),
    }
    logger.info(
        "Dataset loaded: %d samples  |  weather=%s  occupancy=%s",
        len(samples), weather_counts, occupancy_counts,
    )
    return samples, metadata


def process_cnr_ext_patches(detector: YOLOVehicleDetector,
                              samples: List[Dict],
                              batch_size: int = 500) -> Tuple[List[Dict], Dict]:
    results: List[Dict] = []
    correct = 0
    total   = 0
    weather_metrics: Dict[str, Dict] = {}

    logger.info("Starting CNR-EXT inference on %d patches ...", len(samples))

    for i, sample in enumerate(samples):
        try:
            detections = detector.detect(sample["image_path"])
            predicted  = 1 if detections else 0
            max_conf   = max((d.confidence for d in detections), default=0.0)
            gt         = sample["label"]
            is_correct = (predicted == gt)
            if is_correct:
                correct += 1
            total += 1

            w = sample["weather"]
            weather_metrics.setdefault(w, {"correct": 0, "total": 0})
            weather_metrics[w]["total"] += 1
            if is_correct:
                weather_metrics[w]["correct"] += 1

            results.append({
                "image_path":     sample["filename"],
                "ground_truth":   gt,
                "predicted":      predicted,
                "correct":        is_correct,
                "confidence":     max_conf,
                "num_detections": len(detections),
                "weather":        w,
                "camera":         sample["camera"],
                "slot_id":        sample["slot_id"],
            })

            if (i + 1) % batch_size == 0 or i == len(samples) - 1:
                acc = correct / total * 100
                logger.info(
                    "Inference progress: %d/%d  |  running accuracy=%.2f%%",
                    i + 1, len(samples), acc,
                )
        except Exception:
            logger.error(
                "Error on sample %s\n%s",
                sample.get("filename"), traceback.format_exc(),
            )

    tp = sum(1 for r in results if r["predicted"] == 1 and r["ground_truth"] == 1)
    tn = sum(1 for r in results if r["predicted"] == 0 and r["ground_truth"] == 0)
    fp = sum(1 for r in results if r["predicted"] == 1 and r["ground_truth"] == 0)
    fn = sum(1 for r in results if r["predicted"] == 0 and r["ground_truth"] == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / total if total > 0 else 0.0

    for w, d in weather_metrics.items():
        d["accuracy"] = d["correct"] / d["total"] if d["total"] > 0 else 0.0

    metrics = {
        "total_samples":   total,
        "correct":         correct,
        "accuracy":        accuracy,
        "precision":       precision,
        "recall":          recall,
        "f1_score":        f1,
        "true_positives":  tp,
        "true_negatives":  tn,
        "false_positives": fp,
        "false_negatives": fn,
        "weather_metrics": weather_metrics,
    }
    logger.info(
        "Inference complete  |  accuracy=%.2f%%  precision=%.2f%%  "
        "recall=%.2f%%  F1=%.2f%%",
        accuracy * 100, precision * 100, recall * 100, f1 * 100,
    )
    return results, metrics


def save_cnr_ext_results(results: List[Dict], metrics: Dict,
                          metadata: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "cnr_ext_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "dataset":  "CNR-EXT-Patches-150x150",
            "metadata": metadata,
            "metrics":  metrics,
            "sample_results": results[:1000],
        }, f, indent=2)
    logger.info("Detailed results  -> %s", results_path)

    summary_path = output_dir / "cnr_ext_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "dataset":      "CNR-EXT-Patches-150x150",
            "total_samples": metrics["total_samples"],
            "accuracy":      round(metrics["accuracy"]   * 100, 2),
            "precision":     round(metrics["precision"]  * 100, 2),
            "recall":        round(metrics["recall"]     * 100, 2),
            "f1_score":      round(metrics["f1_score"]   * 100, 2),
            "confusion_matrix": {
                "true_positives":  metrics["true_positives"],
                "true_negatives":  metrics["true_negatives"],
                "false_positives": metrics["false_positives"],
                "false_negatives": metrics["false_negatives"],
            },
            "weather_performance": {
                w: {"samples": d["total"], "accuracy": round(d["accuracy"] * 100, 2)}
                for w, d in metrics["weather_metrics"].items()
            },
            "dataset_metadata": metadata,
        }, f, indent=2)
    logger.info("Summary report    -> %s", summary_path)

    csv_path = output_dir / "cnr_ext_predictions.csv"
    with open(csv_path, "w") as f:
        f.write("image,ground_truth,predicted,correct,confidence,weather,camera,slot_id\n")
        for r in results:
            f.write(
                f"{r['image_path']},{r['ground_truth']},{r['predicted']},"
                f"{r['correct']},{r['confidence']:.3f},{r['weather']},"
                f"{r['camera']},{r['slot_id']}\n"
            )
    logger.info("Predictions CSV   -> %s", csv_path)


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("SMART PARKING VIOLATION DETECTION SYSTEM - starting")
    logger.info("Log file: %s", _log_file)
    logger.info("=" * 70)

    output_dir   = Path("results")
    cnr_ext_path = Path("CNR-EXT-Patches-150x150")

    # ── CNR-EXT-Patches mode (primary) ──────────────────────────────────────
    if cnr_ext_path.exists():
        logger.info("[1/4] Loading CNR-EXT-Patches-150x150 dataset ...")
        # Set MAX_SAMPLES to an int (e.g. 1000) for a quick smoke test;
        # set to None to process the full 144,965-image dataset.
        MAX_SAMPLES: Optional[int] = None

        try:
            samples, metadata = load_cnr_ext_patches(cnr_ext_path, max_samples=MAX_SAMPLES)
        except FileNotFoundError:
            logger.critical("Dataset labels file missing.\n%s", traceback.format_exc())
            sys.exit(1)
        except Exception:
            logger.critical("Failed to load dataset.\n%s", traceback.format_exc())
            sys.exit(1)

        logger.info("[2/4] Initialising YOLO detector ...")
        try:
            detector = YOLOVehicleDetector(model_path="yolov8n.pt")
        except Exception:
            logger.critical("YOLO initialisation failed.\n%s", traceback.format_exc())
            sys.exit(1)

        logger.info("[3/4] Running inference ...")
        results, metrics = process_cnr_ext_patches(detector, samples)

        logger.info("[4/4] Saving results ...")
        save_cnr_ext_results(results, metrics, metadata, output_dir)

        logger.info("=" * 70)
        logger.info("CNR-EXT COMPLETE")
        logger.info(
            "  samples=%d  accuracy=%.2f%%  precision=%.2f%%  "
            "recall=%.2f%%  F1=%.2f%%",
            metrics["total_samples"],
            metrics["accuracy"]  * 100,
            metrics["precision"] * 100,
            metrics["recall"]    * 100,
            metrics["f1_score"]  * 100,
        )
        logger.info(
            "  Confusion matrix  TP=%d  TN=%d  FP=%d  FN=%d",
            metrics["true_positives"], metrics["true_negatives"],
            metrics["false_positives"], metrics["false_negatives"],
        )
        for w, d in metrics["weather_metrics"].items():
            logger.info(
                "  %-10s  accuracy=%.2f%%  (%d samples)",
                w, d["accuracy"] * 100, d["total"],
            )
        logger.info("Results in: %s", output_dir.resolve())
        logger.info("=" * 70)
        return detector, results, metrics

    # ── fallback: image-folder / demo mode ──────────────────────────────────
    logger.warning("CNR-EXT-Patches not found – falling back to image-folder mode")
    logger.info("[1/4] Initialising SmartParkingSystem ...")
    try:
        system = SmartParkingSystem()
    except Exception:
        logger.critical("System initialisation failed.\n%s", traceback.format_exc())
        sys.exit(1)

    system.generate_report("results/system_report.json")
    logger.info("  %d slots configured", len(system.slot_polygons))

    image_dir = Path("data/cnrpark/images")
    logger.info("[2/4] Processing images from %s ...", image_dir)

    if image_dir.exists() and any(image_dir.iterdir()):
        results = process_dataset(system, image_dir)
    else:
        logger.warning("No images found – running 5-frame demo with synthetic data")
        results = []
        for frame_id in range(5):
            results.append({
                "frame_id":   frame_id,
                "timestamp":  datetime.now().isoformat(),
                "image_path": f"simulated_frame_{frame_id}.jpg",
                "detections": 3,
                "confirmed_violations": {1: frame_id >= 3, 2: False, 3: frame_id >= 3},
                "occupancy_status": {
                    1: "improper" if frame_id >= 3 else "occupied",
                    2: "empty",
                    3: "improper" if frame_id >= 3 else "occupied",
                },
                "penalties": {1: 25.5, 3: 18.2} if frame_id >= 3 else {},
                "alignment_summary": {
                    1: {"alignment_score": 0.35, "is_improper": True,  "severity": 0.65, "iou": 0.5},
                    3: {"alignment_score": 0.42, "is_improper": True,  "severity": 0.58, "iou": 0.6},
                },
            })

    logger.info("[3/4] Computing statistics ...")
    summary = compute_summary_statistics(results)
    logger.info(
        "  frames=%d  violations=%d  total_penalty=%.2f",
        summary["frames_processed"],
        summary["total_violations_detected"],
        summary["total_penalty_score"],
    )

    logger.info("[4/4] Saving results ...")
    save_results(results, summary, output_dir)

    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info(
        "  frames=%d  violations=%d  unique_slots=%d  total_penalty=%.2f",
        summary["frames_processed"],
        summary["total_violations_detected"],
        summary["unique_slots_with_violations"],
        summary["total_penalty_score"],
    )
    if summary.get("top_violating_slots"):
        for s in summary["top_violating_slots"][:5]:
            logger.info("  slot %s -> %d violations", s["slot_id"], s["violation_count"])
    logger.info("Results in: %s", output_dir.resolve())
    logger.info("=" * 70)
    return system, results, summary


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (KeyboardInterrupt)")
        sys.exit(130)
    except Exception:
        logger.critical("Unhandled exception in main()\n%s", traceback.format_exc())
        sys.exit(1)
