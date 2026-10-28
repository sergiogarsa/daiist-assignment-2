"""Single source of truth for dataset identity, training settings and economic assumptions.

Both train.py and app.py import from here, so every policy (do nothing, review all,
model only, model + review) is always priced with the same numbers.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------------------
# Dataset identity (Roboflow Universe, public, used as provided)
# --------------------------------------------------------------------------------------
RF_WORKSPACE = "udot-classification"
RF_PROJECT = "udot-winter-road-conditions"
RF_VERSION = 1
RF_FORMAT = "folder"
RF_URL = "https://universe.roboflow.com/udot-classification/udot-winter-road-conditions/dataset/1"
RF_LICENSE = "CC BY 4.0"

DATA_DIR = ROOT / "datasets" / f"{RF_PROJECT}-{RF_VERSION}"          # raw download (git-ignored)
SMALL_DIR = ROOT / "datasets" / f"{RF_PROJECT}-{RF_VERSION}-small"   # downscaled working copy
SPLITS = ("train", "valid", "test")
CLASSES = ("Clear", "Slush", "Snow", "Wet")          # Ultralytics sorts folder names alphabetically
POSITIVE_CLASSES = ("Slush", "Snow")                 # "needs treatment now"
NEGATIVE_CLASSES = ("Clear", "Wet")

# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
ART_DIR = ROOT / "artifacts"

# --------------------------------------------------------------------------------------
# Training configuration (CPU-friendly; see REPORT.md)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    model: str = "yolo11n-cls.pt"   # ImageNet-pretrained classification checkpoint
    imgsz: int = 224
    epochs: int = 12
    batch: int = 32
    lr: float = 1e-3                # AdamW peak learning rate (linear warm-up, cosine decay)
    weight_decay: float = 0.01
    seed: int = 0
    deterministic: bool = True
    workers: int = 2
    small_max_side: int = 640       # working-copy resolution (longest side, px)


# --------------------------------------------------------------------------------------
# Economic assumptions (euros, unit = ONE camera check of ONE road segment)
# --------------------------------------------------------------------------------------
@dataclass
class Economics:
    # Outcome costs, expressed as excess cost over a perfect decision ("regret").
    # Treating a road that really is snowy is incurred under every policy (even
    # "do nothing" treats it one cycle later when a patrol finds it), so C_TP = C_TN = 0.
    c_fp: float = 400.0     # wasted spot-treatment sortie (truck, driver call-out, salt)
    c_fn: float = 1000.0    # expected crash + delay cost of one untreated decision cycle (1 h)
    c_tp: float = 0.0
    c_tn: float = 0.0
    prevalence: float = 0.12          # production share of camera checks showing snow/slush
    cases_per_day: int = 1440         # 60 hot-spot cameras x 24 hourly checks
    review_seconds: float = 40.0      # duty officer time per reviewed case
    reviewer_hourly_cost: float = 40.0
    productive_hours_per_reviewer_day: float = 6.0
    review_capacity_hours_per_day: float = 6.0   # spare duty-officer time without new hires
    reviewer_miss_rate: float = 0.05  # P(reviewer says PASS | truly snow/slush)
    reviewer_false_alarm_rate: float = 0.05  # P(reviewer says FAIL | truly clear/wet)

    @property
    def review_cost_per_case(self) -> float:
        return self.review_seconds / 3600.0 * self.reviewer_hourly_cost

    def to_dict(self) -> dict:
        d = asdict(self)
        d["review_cost_per_case"] = self.review_cost_per_case
        return d


ECON = Economics()

# Evidence behind the figures (cited in REPORT.md). Scenario assumptions are labelled as such.
ASSUMPTION_NOTES = {
    "c_fp": "Anchored on Cheshire East's GBP 8.47 per km per gritting run (operational) and Norfolk's "
            "~GBP 49,000 per full run over 49 routes (~GBP 1,000 per ~72 km route). A ~10 km spot "
            "treatment plus ~20 km deadhead => roughly EUR 300-500; EUR 400 used. Scenario estimate.",
    "c_fn": "Scenario build-up for ONE decision cycle (1 h; the next hourly check is a new case, so "
            "longer exposure is not double counted): 10 km segment x 600 veh/h = 6,000 veh-km; +3 crashes per "
            "million veh-km when snow/slush is untreated (between the ~+20% injury-crash effect of reduced "
            "salting in Finland and the ~8x before/after-salting ratio in Kuemmel & Hanbali 1992) => 0.018 "
            "crashes x EUR 40,000 average crash (EU external-cost handbook injury values mixed with "
            "property-damage-only crashes) = EUR 720; + ~EUR 150 delay (FHWA: arterial speeds fall 30-40% on "
            "snowy/slushy pavement); + ~EUR 130 extra de-icing once snow is compacted. Total ~EUR 1,000. "
            "Most uncertain number; see sensitivity and break-even.",
    "prevalence": "Scenario assumption for winter-season operations; the dataset is ~71% snow/slush, "
                  "which is curated, not a production rate.",
    "cases_per_day": "Scenario: 60 hot-spot cameras checked hourly, 24 h/day in winter.",
    "review_seconds": "Scenario: open the image, glance at the neighbouring camera and pavement sensor.",
    "reviewer_hourly_cost": "Eurostat 2024 average hourly labour cost EUR 37.3 (euro area), rounded "
                            "up for 24/7 shift premia.",
    "productive_hours_per_reviewer_day": "Scenario: 6 of 8 shift hours available for reviewing.",
    "review_capacity_hours_per_day": "Scenario: three 8 h duty-officer shifts can each spare ~2 h.",
    "reviewer_error": "Scenario: 5% miss / 5% false alarm; perfect (0/0) and 10/10 tested as alternatives.",
}
