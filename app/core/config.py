import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
SAMPLES_DIR = STATIC_DIR / "samples"
THUMBNAILS_DIR = STATIC_DIR / "thumbnails"
CLIPS_DIR = STATIC_DIR / "clips"

# Ensure directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Pretrained Hugging Face Model
MODEL_NAME = "OPear/videomae-large-finetuned-UCF-Crime"

# UCF-Crime 14 Classes
CLASS_MAPPING = {
    "Abuse": 0,
    "Arrest": 1,
    "Arson": 2,
    "Assault": 3,
    "Burglary": 4,
    "Explosion": 5,
    "Fighting": 6,
    "Normal Videos": 7,
    "Road Accidents": 8,
    "Robbery": 9,
    "Shooting": 10,
    "Shoplifting": 11,
    "Stealing": 12,
    "Vandalism": 13,
}

REVERSE_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
NORMAL_CLASS_ID = 7
NORMAL_LABEL = "Normal Videos"
NORMAL_LABELS = {NORMAL_LABEL, "Normal_Videos_event"}

# Severity ratings for triage
SEVERITY_MAPPING = {
    "Shooting": "CRITICAL",
    "Explosion": "CRITICAL",
    "Assault": "HIGH",
    "Fighting": "HIGH",
    "Arson": "HIGH",
    "Robbery": "HIGH",
    "Burglary": "MEDIUM",
    "Stealing": "MEDIUM",
    "Shoplifting": "MEDIUM",
    "Vandalism": "MEDIUM",
    "Road Accidents": "HIGH",
    "RoadAccidents": "HIGH",
    "Abuse": "HIGH",
    "Arrest": "LOW",
    "Normal Videos": "NONE",
    "Normal_Videos_event": "NONE",
}

# Color palette for crime categories in UI
CATEGORY_COLORS = {
    "Normal Videos": "#10b981",    # Emerald
    "Normal_Videos_event": "#10b981",
    "Shooting": "#ef4444",         # Red
    "Explosion": "#f97316",        # Orange
    "Assault": "#f43f5e",         # Rose
    "Fighting": "#e11d48",        # Crimson
    "Arson": "#ea580c",           # Dark Orange
    "Robbery": "#d946ef",         # Fuchsia
    "Burglary": "#a855f7",        # Purple
    "Stealing": "#8b5cf6",        # Violet
    "Shoplifting": "#6366f1",     # Indigo
    "Vandalism": "#06b6d4",       # Cyan
    "Road Accidents": "#eab308",   # Amber
    "RoadAccidents": "#eab308",
    "Abuse": "#ec4899",           # Pink
    "Arrest": "#3b82f6",          # Blue
}

# VideoMAE Model Specifications
NUM_FRAMES_PER_WINDOW = 16
FRAME_INPUT_SIZE = (224, 224)
DEFAULT_WINDOW_SECONDS = 2.5
DEFAULT_STRIDE_SECONDS = 1.0
ANOMALY_THRESHOLD = 0.50  # Anomaly score (1 - P(Normal)) threshold for anomaly detection
PEAK_CONFIDENCE_THRESHOLD = 0.60

# If False: analysis endpoints return HTTP 503 when VideoMAE is not loaded.
# If True: the heuristic kinetic engine may run, but results will be tagged engine_mode=heuristic.
ALLOW_HEURISTIC_FALLBACK = False

# Stealing-Specific Object Detection & Tracking Configuration
YOLO_WEIGHTS_PATH = BASE_DIR / "app" / "core" / "weights" / "yolov8n.pt"
STEALING_ANALYSIS_ENABLED = True
STEALING_CANDIDATE_THRESHOLD = 0.30  # Candidate threshold to trigger object analysis
STEALING_CONTEXT_BUFFER_SECONDS = 1.0  # Context padding (seconds) before & after candidate clip
STEALING_FRAME_SAMPLE_FPS = 5.0  # Frame sampling rate for object tracking (optimizes CPU latency)
STEALING_DETECTION_CONF_THRESHOLD = 0.25  # Minimum detection confidence
STEALING_PERSON_CLASS_IDS = [0]  # COCO class 0: person
STEALING_TARGET_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    67: "cell phone",
}  # Extensible target object classes
STEALING_PROXIMITY_THRESHOLD = 0.30  # Normalized Euclidean distance threshold for person-object proximity
STEALING_DISPLACEMENT_THRESHOLD = 0.08  # Normalized displacement threshold for object position change

# Dedicated Weapon Detection & Generalized Multi-Object Tracking Configuration
WEAPON_WEIGHTS_PATH = BASE_DIR / "app" / "core" / "weights" / "yolov8_weapon.pt"
WEAPON_DETECTION_ENABLED = True
WEAPON_CONFIDENCE_THRESHOLD = 0.45
WEAPON_CLASSES = {0: "Pistol", 1: "Rifle", 2: "Knife"}
WEAPON_PROXIMITY_THRESHOLD = 0.32  # Normalized distance for person-weapon proximity cue
TRACKING_DEFAULT_FPS = 10.0  # Default sampling rate for generalized tracking lab and incident review

# Calibrated class-specific confidence thresholds for false-positive reduction
# Evaluated on positive fixtures and negative surveillance footage:
# - Weapons require high confidence (0.40 - 0.50) to prevent false alerts on hands, clothing folds, car door handles, and shadows.
# - High-frequency hallucination COCO classes (clock on hubcaps, umbrella on bent arms, skateboard on shadows) require elevated thresholds (0.45 - 0.50).
# - Standard robust classes (person, car, truck, bus, motorcycle, bicycle) operate reliably at 0.35.
CLASS_CONFIDENCE_THRESHOLDS = {
    # Weapons (WeaponShield)
    "Pistol": 0.50,
    "Rifle": 0.45,
    "Knife": 0.40,
    # Common CCTV False-Positive Hallucination Classes
    "clock": 0.50,       # Hubcaps and wheel rims
    "umbrella": 0.45,    # Bent arms and dark rain jackets
    "skateboard": 0.45,  # Vehicle shadows, curbs, ground marks
    "kite": 0.50,        # Overhead street fixtures, wires
    "toaster": 0.50,
    "hair drier": 0.50,
    "handbag": 0.38,     # Clothing pockets, shadows
    "cell phone": 0.38,  # Hands, small wallets
    "backpack": 0.35,
    "suitcase": 0.35,
    # High-data robust classes
    "person": 0.35,
    "car": 0.35,
    "motorcycle": 0.35,
    "bicycle": 0.35,
    "bus": 0.35,
    "truck": 0.35,
}
DEFAULT_CLASS_CONFIDENCE_THRESHOLD = 0.35

# Temporal consistency and appearance interval settings
TRACK_CONFIRMATION_MIN_OBSERVATIONS = 2  # Minimum observed frames to designate a track as 'confirmed_track' vs 'uncertain_candidate'
APPEARANCE_INTERVAL_MAX_GAP_SECONDS = 1.5  # Max temporal gap between consecutive observations before splitting appearance intervals



