"""
Filters a yago entity json file down to only entities that belong to one of
a given list of subclasses. "Belongs to" is checked against the entity's
source_class / source_class_iri only.

An entity is included if its source_class (local_name or iri) matches ANY
of the target classes.

Usage:
    python filter_by_subclasses.py
(edit TARGET_CLASSES and the paths below)
"""

import json


def entity_matches(entity: dict, target_classes: set) -> bool:
    # Check source_class (local name) and source_class_iri only
    if entity.get("source_class") in target_classes:
        return True

    return False


def filter_by_subclasses(input_path: str, output_path: str, target_classes) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_set = set(target_classes)

    filtered = {
        iri: entity
        for iri, entity in data.items()
        if entity_matches(entity, target_set)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Input entries:    {len(data)}")
    print(f"Target classes:   {sorted(target_set)}")
    print(f"Matched entries:  {len(filtered)}")
    print(f"Saved to: {output_path}")

def filter_by_depth(input_path: str, output_path: str, target_depth: int) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = {
        iri: entity
        for iri, entity in data.items()
        if entity.get("depth") == target_depth
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Input entries:    {len(data)}")
    print(f"Target depth:     {target_depth}")
    print(f"Matched entries:  {len(filtered)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    INPUT_PATH = "entity_registry_fixed.json"
    OUTPUT_PATH = "comp_player_training_data.json"

    TARGET_CLASSES = [
    "Australian_Rules_Football_Player_Q13414980",
    "Badminton_player",
    "Baseball_player",
    "Basketball_player",
    "Bridge_Player_Q18437198",
    "Cricketer",
    "Cue_Sports_Player_Q18544928",
    "Curler_Q17516936",
    "Field_Hockey_Player_Q10843263",
    "Futsal_Player_Q18515558",
    "Gaelic_Football_Player_Q17351861",
    "Gamer",
    "Golfer",
    "Gridiron_football_player",
    "Handball_player",
    "Ice_hockey_player",
    "Lacrosse_Player_Q17682262",
    "Netballer_Q17619498",
    "Poker_Player_Q15295720",
    "Rugby_Player_Q13415036",
    "Softball_Player_Q13388586",
    "Squash_Player_Q16278103",
    "Table_Tennis_Player_Q13382519",
    "Tennis_player",
    "Volleyball_player",
    "Water_Polo_Player_Q17524364",
]
    print(f"Target classes: {len(TARGET_CLASSES)}")

    DEPTH = -1

    filter_by_subclasses(INPUT_PATH, OUTPUT_PATH, TARGET_CLASSES)
    #filter_by_depth(INPUT_PATH, OUTPUT_PATH, DEPTH)