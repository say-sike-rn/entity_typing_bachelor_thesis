"""
Filters entities from a bertopic-labeled CSV file down to only those
belonging to a chosen list of bertopic topics, and writes them out as a
JSON file keyed by IRI (dict of iri -> entity), matching your usual format
but limited to what's actually needed for training: local_name, description,
and bertopic_topic (used as the label).

Entities with bertopic_topic == -1 (BERTopic's "no topic assigned" / outlier
bucket) are always excluded, since -1 isn't a real topic.

Usage:
    python filter_by_bertopic_topic.py
(edit INPUT_CSV_PATH, OUTPUT_JSON_PATH, and TARGET_TOPICS below)
"""

import csv
import json


def filter_by_bertopic_topic(input_csv_path: str, output_json_path: str, target_topics) -> None:
    target_set = set(target_topics)

    filtered = {}
    total_rows = 0
    excluded_outlier = 0
    excluded_not_target = 0

    with open(input_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            total_rows += 1

            topic_str = row.get("bertopic_topic", "").strip()
            try:
                topic = int(topic_str)
            except (ValueError, TypeError):
                excluded_not_target += 1
                continue

            if topic == -1:
                excluded_outlier += 1
                continue

            if topic not in target_set:
                excluded_not_target += 1
                continue

            iri = row["iri"]
            filtered[iri] = {
                "local_name": row.get("local_name", ""),
                "description": row.get("description", ""),
                "bertopic_topic": topic,
            }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"Total rows read:              {total_rows}")
    print(f"Excluded (topic == -1):       {excluded_outlier}")
    print(f"Excluded (not in target set): {excluded_not_target}")
    print(f"Matched entries:              {len(filtered)}")
    print(f"Target topics:                {sorted(target_set)}")
    print(f"Saved to: {output_json_path}")


if __name__ == "__main__":
    INPUT_CSV_PATH = "../../topics/bertopic_output_150/entities_with_topics.csv"
    OUTPUT_JSON_PATH = "../../training/topics_model1_trainingdata.json"

    # List the bertopic topic numbers you want to keep.
    TARGET_TOPICS = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        # add more topic numbers here
    ]

    filter_by_bertopic_topic(INPUT_CSV_PATH, OUTPUT_JSON_PATH, TARGET_TOPICS)