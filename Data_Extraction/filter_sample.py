"""
Removes entities present in a sample file (e.g. 1000_sample.json) from the
full 40k entity dataset, producing a new json file with the remainder.

Usage:
    python filter_out_sample.py
(edit the paths below, or adapt to argparse if you prefer CLI args)
"""

import json
from pathlib import Path


def remove_sampled_entities(full_path: str, sample_path: str, output_path: str) -> None:
    with open(full_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    with open(sample_path, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    sample_keys = set(sample_data.keys())

    remaining = {
        iri: entity
        for iri, entity in full_data.items()
        if iri not in sample_keys
    }

    removed_count = len(full_data) - len(remaining)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(remaining, f, ensure_ascii=False, indent=2)

    print(f"Full dataset entries:   {len(full_data)}")
    print(f"Sample entries:         {len(sample_data)}")
    print(f"Removed (found in sample): {removed_count}")
    print(f"Remaining entries:      {len(remaining)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    FULL_DATASET_PATH = "layers_entity_registry.json"
    SAMPLE_PATH = "layers_sample_1000.json"
    OUTPUT_PATH = "layers_entities_without_sample.json"

    remove_sampled_entities(FULL_DATASET_PATH, SAMPLE_PATH, OUTPUT_PATH)