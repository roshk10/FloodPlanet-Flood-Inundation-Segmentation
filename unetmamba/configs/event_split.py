from pathlib import Path
import json
import random


# ============================================================
# FloodPlanet fixed event-level split
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "FloodPlanet"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "configs" / "event_split.json"
)

SEED = 2026

TRAIN_EVENTS = 14
VAL_EVENTS = 2
TEST_EVENTS = 3


def find_event_directories():

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_ROOT}"
        )

    events = sorted(
        p.name
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
    )

    if len(events) != 19:
        raise RuntimeError(
            f"Expected 19 FloodPlanet events, "
            f"found {len(events)}"
        )

    return events


def main():

    events = find_event_directories()

    rng = random.Random(SEED)

    shuffled = events.copy()
    rng.shuffle(shuffled)

    test_events = sorted(
        shuffled[:TEST_EVENTS]
    )

    val_events = sorted(
        shuffled[
            TEST_EVENTS:
            TEST_EVENTS + VAL_EVENTS
        ]
    )

    train_events = sorted(
        shuffled[
            TEST_EVENTS + VAL_EVENTS:
        ]
    )

    split = {
        "seed": SEED,
        "dataset": "FloodPlanet",
        "protocol": "fixed event-level split",
        "train": train_events,
        "validation": val_events,
        "test": test_events
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w"
    ) as f:

        json.dump(
            split,
            f,
            indent=4
        )

    print("=" * 70)
    print("FLOODPLANET EVENT SPLIT")
    print("=" * 70)

    print(
        f"\nTrain events ({len(train_events)}):"
    )

    for event in train_events:
        print("  ", event)

    print(
        f"\nValidation events ({len(val_events)}):"
    )

    for event in val_events:
        print("  ", event)

    print(
        f"\nTest events ({len(test_events)}):"
    )

    for event in test_events:
        print("  ", event)

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
