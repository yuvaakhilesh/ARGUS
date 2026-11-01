"""
ARGUS Dynamic Object Catalog.

Combines supported classes from all active models:
- General Object Detector: Ultralytics YOLOv8n (80 COCO classes)
- Dedicated Weapon Detector: Ultralytics YOLOv8s WeaponShield (Pistol, Rifle, Knife)

Provides structured taxonomy, functional categorization, search utilities,
and inventory aggregation for forensic tracking review.
"""

from typing import Any, Dict, List, Optional, Set

# Functional category groupings for COCO 80 classes + Weapon classes
CLASS_CATEGORIES: Dict[str, List[str]] = {
    "People": [
        "person",
    ],
    "Weapons": [
        "Pistol",
        "Rifle",
        "Knife",
    ],
    "Personal Belongings": [
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "cell phone",
    ],
    "Vehicles": [
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    ],
    "Outdoor & Street": [
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
    ],
    "Animals": [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    ],
    "Sports Equipment": [
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
    ],
    "Kitchen & Dining": [
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",  # COCO generic knife
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
    ],
    "Furniture & Interior": [
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
    ],
    "Electronics & Appliances": [
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
    ],
    "Household & Accessories": [
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ],
}

# Category colors for UI badges and bounding box styling
CATEGORY_COLORS: Dict[str, str] = {
    "People": "#06b6d4",               # Cyan
    "Weapons": "#ef4444",              # Crimson / Red
    "Personal Belongings": "#a855f7",  # Purple
    "Vehicles": "#f59e0b",             # Amber
    "Outdoor & Street": "#64748b",     # Slate
    "Animals": "#10b981",              # Emerald
    "Sports Equipment": "#3b82f6",     # Blue
    "Kitchen & Dining": "#ec4899",     # Pink
    "Furniture & Interior": "#8b5cf6", # Violet
    "Electronics & Appliances": "#0284c7", # Sky Blue
    "Household & Accessories": "#6b7280",  # Gray
}

# Comprehensive natural language alias mapping to canonical model class names
CLASS_ALIASES: Dict[str, List[str]] = {
    # Weapons & Firearms
    "handgun": ["Pistol"],
    "glock": ["Pistol"],
    "revolver": ["Pistol"],
    "pistol": ["Pistol"],
    "firearm": ["Pistol", "Rifle"],
    "firearms": ["Pistol", "Rifle"],
    "gun": ["Pistol", "Rifle"],
    "guns": ["Pistol", "Rifle"],
    "rifle": ["Rifle"],
    "longgun": ["Rifle"],
    "shotgun": ["Rifle"],
    "assault rifle": ["Rifle"],
    "carbine": ["Rifle"],
    "knife": ["Knife"],
    "knives": ["Knife"],
    "blade": ["Knife"],
    "dagger": ["Knife"],
    "machete": ["Knife"],
    "sword": ["Knife"],
    "weapon": ["Pistol", "Rifle", "Knife"],
    "weapons": ["Pistol", "Rifle", "Knife"],

    # People
    "person": ["person"],
    "people": ["person"],
    "pedestrian": ["person"],
    "human": ["person"],
    "man": ["person"],
    "woman": ["person"],
    "boy": ["person"],
    "girl": ["person"],
    "guy": ["person"],
    "lady": ["person"],
    "individual": ["person"],
    "crowd": ["person"],

    # Personal Belongings
    "mobile": ["cell phone"],
    "phone": ["cell phone"],
    "cellphone": ["cell phone"],
    "cell phone": ["cell phone"],
    "cell": ["cell phone"],
    "smartphone": ["cell phone"],
    "telephone": ["cell phone"],
    "bag": ["backpack", "handbag", "suitcase"],
    "bags": ["backpack", "handbag", "suitcase"],
    "backpack": ["backpack"],
    "backpacks": ["backpack"],
    "rucksack": ["backpack"],
    "handbag": ["handbag"],
    "purse": ["handbag"],
    "tote": ["handbag"],
    "suitcase": ["suitcase"],
    "luggage": ["suitcase"],
    "baggage": ["suitcase"],
    "briefcase": ["suitcase"],
    "umbrella": ["umbrella"],
    "parasol": ["umbrella"],
    "brolly": ["umbrella"],
    "tie": ["tie"],
    "necktie": ["tie"],

    # Vehicles
    "car": ["car"],
    "cars": ["car"],
    "auto": ["car"],
    "automobile": ["car"],
    "sedan": ["car"],
    "suv": ["car"],
    "van": ["car"],
    "vehicle": ["car", "motorcycle", "bus", "truck", "bicycle"],
    "vehicles": ["car", "motorcycle", "bus", "truck", "bicycle"],
    "bike": ["bicycle", "motorcycle"],
    "bikes": ["bicycle", "motorcycle"],
    "bicycle": ["bicycle"],
    "cycle": ["bicycle"],
    "motorcycle": ["motorcycle"],
    "motorbike": ["motorcycle"],
    "scooter": ["motorcycle"],
    "moped": ["motorcycle"],
    "bus": ["bus"],
    "truck": ["truck"],
    "lorry": ["truck"],
    "pickup": ["truck"],
    "airplane": ["airplane"],
    "plane": ["airplane"],
    "flight": ["airplane"],
    "train": ["train"],
    "locomotive": ["train"],
    "boat": ["boat"],
    "ship": ["boat"],
    "vessel": ["boat"],

    # Electronics & Appliances
    "tv": ["tv"],
    "television": ["tv"],
    "monitor": ["tv"],
    "screen": ["tv"],
    "display": ["tv"],
    "laptop": ["laptop"],
    "notebook": ["laptop"],
    "computer": ["laptop"],
    "pc": ["laptop"],
    "mouse": ["mouse"],
    "remote": ["remote"],
    "remote control": ["remote"],
    "keyboard": ["keyboard"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "toaster": ["toaster"],
    "sink": ["sink"],
    "refrigerator": ["refrigerator"],
    "fridge": ["refrigerator"],

    # Kitchen & Dining
    "bottle": ["bottle"],
    "water bottle": ["bottle"],
    "glass": ["wine glass"],
    "wine glass": ["wine glass"],
    "cup": ["cup"],
    "mug": ["cup"],
    "fork": ["fork"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],

    # Furniture & Interior
    "chair": ["chair"],
    "seat": ["chair"],
    "couch": ["couch"],
    "sofa": ["couch"],
    "bed": ["bed"],
    "dining table": ["dining table"],
    "table": ["dining table"],
    "toilet": ["toilet"],
    "potted plant": ["potted plant"],
    "plant": ["potted plant"],

    # Animals
    "dog": ["dog"],
    "puppy": ["dog"],
    "pup": ["dog"],
    "canine": ["dog"],
    "hound": ["dog"],
    "cat": ["cat"],
    "kitten": ["cat"],
    "feline": ["cat"],
    "bird": ["bird"],
    "horse": ["horse"],
    "sheep": ["sheep"],
    "cow": ["cow"],
    "elephant": ["elephant"],
    "bear": ["bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],

    # Sports
    "skateboard": ["skateboard"],
    "skate": ["skateboard"],
    "ball": ["sports ball"],
    "sports ball": ["sports ball"],
    "frisbee": ["frisbee"],
    "skis": ["skis"],
    "snowboard": ["snowboard"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat"],
    "bat": ["baseball bat"],
    "tennis racket": ["tennis racket"],
    "racket": ["tennis racket"],

    # Household
    "clock": ["clock"],
    "watch": ["clock"],
    "book": ["book"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear"],
    "hair drier": ["hair drier"],
    "toothbrush": ["toothbrush"],
}


class ObjectCatalog:
    """
    Catalog manager that introspects active detectors and exposes
    the combined class vocabulary, taxonomy, and search utilities.
    """

    @classmethod
    def get_supported_classes(cls) -> List[Dict[str, Any]]:
        """
        Dynamically extracts supported classes from active detection models.
        Returns unified catalog entries with metadata.
        """
        catalog: List[Dict[str, Any]] = []

        # 1. Dedicated Weapon Detector classes
        from app.core.weapon_detector import WeaponDetector
        w_detector = WeaponDetector.get_instance()
        weapon_classes = getattr(w_detector, "supported_classes", {0: "Pistol", 1: "Rifle", 2: "Knife"})
        for cid, name in weapon_classes.items():
            catalog.append({
                "class_name": name,
                "normalized_name": name.lower(),
                "class_id": int(cid),
                "model_source": "weapon_shield_yolov8s",
                "model_name": "yolov8_weapon.pt",
                "category": "Weapons",
                "color": CATEGORY_COLORS["Weapons"],
                "is_weapon": True,
                "is_person": False,
                "is_supported": True,
                "description": f"Dedicated tactical weapon detection ({name})",
            })

        # 2. General YOLOv8n detector classes (80 COCO classes)
        from app.core.stealing_detector import StealingObjectDetector
        g_detector = StealingObjectDetector.get_instance()
        if not g_detector.is_loaded:
            g_detector.load_model()
        coco_classes = getattr(g_detector, "supported_classes", {})

        for cid, name in coco_classes.items():
            cat = cls._find_category(name)
            color = CATEGORY_COLORS.get(cat, "#94a3b8")
            is_person = (int(cid) == 0 or name.lower() == "person")
            catalog.append({
                "class_name": name,
                "normalized_name": name.lower(),
                "class_id": int(cid),
                "model_source": "coco_yolov8n",
                "model_name": "yolov8n.pt",
                "category": cat,
                "color": color,
                "is_weapon": False,
                "is_person": is_person,
                "is_supported": True,
                "description": f"COCO General Object ({name})",
            })

        return catalog

    @classmethod
    def _find_category(cls, class_name: str) -> str:
        """Finds matching category group for a class name."""
        name_lower = class_name.lower()
        for cat, items in CLASS_CATEGORIES.items():
            if any(item.lower() == name_lower for item in items):
                return cat
        return "Common Objects"

    @classmethod
    def get_class_threshold(cls, class_name: str) -> float:
        """Returns the calibrated confidence threshold for the specified class."""
        from app.core.config import CLASS_CONFIDENCE_THRESHOLDS, DEFAULT_CLASS_CONFIDENCE_THRESHOLD
        name_lower = class_name.lower()
        for k, v in CLASS_CONFIDENCE_THRESHOLDS.items():
            if k.lower() == name_lower:
                return v
        return DEFAULT_CLASS_CONFIDENCE_THRESHOLD

    @classmethod
    def resolve_query(cls, query: str) -> Dict[str, Any]:
        """
        Resolves a user-typed query string across canonical classes and aliases.
        Distinguishes exact matches, alias mappings, partial matches, and unsupported classes.
        """
        q = (query or "").strip().lower()
        if not q:
            return {
                "query": query,
                "is_supported": False,
                "canonical_classes": [],
                "matched_aliases": [],
                "explanation": "Query is empty.",
            }

        all_classes = cls.get_supported_classes()
        class_by_name = {c["class_name"].lower(): c["class_name"] for c in all_classes}

        matched_canonicals = set()
        matched_aliases = []

        # 1. Exact alias match
        if q in CLASS_ALIASES:
            for c_alias in CLASS_ALIASES[q]:
                if c_alias.lower() in class_by_name:
                    matched_canonicals.add(class_by_name[c_alias.lower()])
            matched_aliases.append(q)

        # 2. Exact class name match
        if q in class_by_name:
            matched_canonicals.add(class_by_name[q])

        # 3. Partial alias & partial class match only if no exact match found
        if not matched_canonicals:
            q_words = set(q.split())
            for alias, targets in CLASS_ALIASES.items():
                if (q in alias or alias in q_words) and alias != q:
                    for t in targets:
                        if t.lower() in class_by_name:
                            matched_canonicals.add(class_by_name[t.lower()])
                    matched_aliases.append(alias)

            for c_lower, c_orig in class_by_name.items():
                if (q in c_lower or c_lower in q_words) and c_orig not in matched_canonicals:
                    matched_canonicals.add(c_orig)

            for cat_name, cat_items in CLASS_CATEGORIES.items():
                if q == cat_name.lower() or cat_name.lower() in q_words:
                    for item in cat_items:
                        if item.lower() in class_by_name:
                            matched_canonicals.add(class_by_name[item.lower()])

        canonical_list = sorted(list(matched_canonicals))
        is_supported = len(canonical_list) > 0

        # Construct helpful explanation
        if is_supported:
            if q in ["gun", "guns", "firearm", "firearms"]:
                explanation = f"Query '{query}' resolved to supported firearm classes: {', '.join(canonical_list)}."
            elif q in ["handgun", "glock", "revolver"]:
                explanation = f"Query '{query}' resolved to dedicated weapon class: Pistol."
            elif q in ["mobile", "phone", "smartphone"]:
                explanation = f"Query '{query}' resolved to standard class: cell phone."
            elif q in ["knife", "knives", "blade", "dagger"]:
                explanation = f"Query '{query}' resolved to tactical weapon detector: Knife (with COCO knife deduplication)."
            elif len(canonical_list) == 1:
                explanation = f"Query '{query}' matched class: {canonical_list[0]}."
            else:
                explanation = f"Query '{query}' matched {len(canonical_list)} supported classes: {', '.join(canonical_list)}."
        else:
            explanation = f"Class '{query}' is not supported by active models (YOLOv8 COCO 80 + WeaponShield 3)."

        return {
            "query": query,
            "is_supported": is_supported,
            "canonical_classes": canonical_list,
            "matched_aliases": list(set(matched_aliases))[:5],
            "explanation": explanation,
        }

    @classmethod
    def search_catalog(cls, query: str) -> Dict[str, Any]:
        """
        Searches the model catalog by query string using alias resolution.
        Returns matched supported classes or explicit unsupported status.
        """
        q = (query or "").strip().lower()
        if not q:
            return {
                "query": query,
                "matched_classes": [],
                "is_supported": False,
                "message": "Empty search query.",
            }

        resolved = cls.resolve_query(q)
        all_classes = cls.get_supported_classes()
        canonical_set = set(resolved["canonical_classes"])

        matched_items = [c for c in all_classes if c["class_name"] in canonical_set]

        return {
            "query": query,
            "is_supported": resolved["is_supported"],
            "matched_count": len(matched_items),
            "matched_classes": matched_items,
            "canonical_classes": resolved["canonical_classes"],
            "matched_aliases": resolved["matched_aliases"],
            "explanation": resolved["explanation"],
            "message": resolved["explanation"],
        }

    @classmethod
    def build_inventory(cls, track_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes accurate object inventory statistics from actual track summaries.
        Returns class-level breakdown with total detections and unique track counts.
        """
        inventory: Dict[str, Dict[str, Any]] = {}
        category_counts: Dict[str, int] = {}

        for track in track_summaries:
            cname = track.get("class_name", "Unknown")
            det_count = int(track.get("detection_count", 0))
            is_wep = bool(track.get("is_weapon"))
            is_per = bool(track.get("is_person"))
            model_src = track.get("model_source", "weapon_shield_yolov8s" if is_wep else "coco_yolov8n")
            cat = "Weapons" if is_wep else cls._find_category(cname)

            if cname not in inventory:
                inventory[cname] = {
                    "class_name": cname,
                    "category": cat,
                    "color": CATEGORY_COLORS.get(cat, "#94a3b8"),
                    "model_source": model_src,
                    "is_weapon": is_wep,
                    "is_person": is_per,
                    "unique_tracks": 0,
                    "total_detections": 0,
                    "max_confidence": 0.0,
                    "first_seen_s": float("inf"),
                    "last_seen_s": float("-inf"),
                    "track_ids": [],
                }

            inv_entry = inventory[cname]
            inv_entry["unique_tracks"] += 1
            inv_entry["total_detections"] += det_count
            inv_entry["max_confidence"] = max(inv_entry["max_confidence"], float(track.get("max_confidence", 0.0)))
            inv_entry["first_seen_s"] = min(inv_entry["first_seen_s"], float(track.get("first_seen_s", 0.0)))
            inv_entry["last_seen_s"] = max(inv_entry["last_seen_s"], float(track.get("last_seen_s", 0.0)))
            inv_entry["track_ids"].append(track.get("track_id_str", f"ID-{track.get('track_id')}"))

            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Format rows sorted by detections descending
        rows = list(inventory.values())
        rows.sort(key=lambda r: (r["is_weapon"], r["total_detections"]), reverse=True)
        for r in rows:
            if r["first_seen_s"] == float("inf"):
                r["first_seen_s"] = 0.0
            if r["last_seen_s"] == float("-inf"):
                r["last_seen_s"] = 0.0
            r["max_confidence"] = round(r["max_confidence"], 4)

        return {
            "total_objects_tracked": len(track_summaries),
            "distinct_classes_detected": len(rows),
            "category_summary": category_counts,
            "inventory": rows,
        }

