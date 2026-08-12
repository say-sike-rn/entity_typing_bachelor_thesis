"""
Stratified-like sampling of YAGO entities by direct_parents, depth, and description length.
(Its not by definition stratified but behaves very similarly.)

Strategy:
1. Expand: each entity counts toward the quota of EVERY direct_parent it has.
2. Balance across parents: aim for a roughly equal target-per-parent, with a
   floor (min per parent, if available) and a cap (so no single parent hogs
   the budget) — implemented via a "largest remainder" style water-filling.
3. Within each parent's allocation, balance further across depth and length
   bucket as evenly as possible, given availability.
4. Dedupe: once an entity is selected, it's marked used and won't be
   re-picked when filling a different parent's quota — but it still counts
   toward that other parent's fulfilled count.
"""

import json
import random
from collections import defaultdict

random.seed(42)

INPUT_PATH = "layers_entity_registry.json"   
OUTPUT_PATH = "layers_sample_1000.json"
TARGET_TOTAL = 1000

SHORT_MAX = 400
MEDIUM_MAX = 700


def length_bucket(desc: str) -> str:
    n = len(desc)
    if n < SHORT_MAX:
        return "short"
    elif n <= MEDIUM_MAX:
        return "medium"
    else:
        return "long"


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(data):
    """
    Returns:
      entities: dict iri -> record (with added 'length_bucket')
      parent_to_entities: dict parent_local_name -> list of iris
    """
    entities = {}
    parent_to_entities = defaultdict(list)

    for iri, rec in data.items():
        desc = rec.get("description", "") or ""
        rec = dict(rec)  # shallow copy so we don't mutate original
        rec["length_bucket"] = length_bucket(desc)
        entities[iri] = rec

        parents = rec.get("direct_parents", [])
        if not parents:
            parent_to_entities["__NO_PARENT__"].append(iri)
        else:
            for p in parents:
                parent_to_entities[p["local_name"]].append(iri)

    return entities, parent_to_entities


def allocate_targets(parent_to_entities, target_total):
    """
    Flat/balanced allocation across parents using water-filling:
    start with equal share, redistribute unused capacity from parents
    that have fewer available entities than their share.
    """
    parents = list(parent_to_entities.keys())
    n_parents = len(parents)
    availability = {p: len(v) for p, v in parent_to_entities.items()}

    remaining_target = target_total
    remaining_parents = set(parents)
    allocation = {p: 0 for p in parents}

    # Iteratively distribute equally, capping at availability, until stable
    while remaining_parents and remaining_target > 0:
        share = remaining_target / len(remaining_parents)
        exhausted = []
        for p in list(remaining_parents):
            cap = availability[p] - allocation[p]
            if cap <= share:
                allocation[p] += cap
                remaining_target -= cap
                exhausted.append(p)
        if exhausted:
            for p in exhausted:
                remaining_parents.discard(p)
        else:
            # even split of remaining_target across remaining_parents
            base = int(share)
            for p in remaining_parents:
                allocation[p] += base
                remaining_target -= base
            # leftover due to rounding -> assign one-by-one
            leftover_parents = list(remaining_parents)
            random.shuffle(leftover_parents)
            i = 0
            while remaining_target > 0 and leftover_parents:
                p = leftover_parents[i % len(leftover_parents)]
                if allocation[p] < availability[p]:
                    allocation[p] += 1
                    remaining_target -= 1
                i += 1
                if i > 10000:  # safety
                    break
            break

    return allocation, availability


def balanced_pick_within_parent(candidate_iris, entities, n_pick, used):
    """
    Given candidate entity iris for one parent, pick n_pick of them,
    trying to balance across (depth, length_bucket) combos as evenly
    as possible, while respecting the global `used` set (dedup).
    """
    available = [iri for iri in candidate_iris if iri not in used]
    if n_pick <= 0 or not available:
        return []

    # group by (depth, length_bucket)
    groups = defaultdict(list)
    for iri in available:
        rec = entities[iri]
        key = (rec.get("depth"), rec["length_bucket"])
        groups[key].append(iri)

    for g in groups.values():
        random.shuffle(g)

    group_keys = list(groups.keys())
    picked = []
    # round-robin across groups until n_pick reached or exhausted
    idx = 0
    exhausted = set()
    while len(picked) < n_pick and len(exhausted) < len(group_keys):
        key = group_keys[idx % len(group_keys)]
        if key not in exhausted:
            if groups[key]:
                picked.append(groups[key].pop())
            else:
                exhausted.add(key)
        idx += 1

    return picked


def main():
    data = load_data(INPUT_PATH)
    entities, parent_to_entities = build_index(data)

    print(f"Loaded {len(entities)} entities across {len(parent_to_entities)} distinct direct_parents")

    allocation, availability = allocate_targets(parent_to_entities, TARGET_TOTAL)

    used = set()
    selected = []

    # process parents in order of scarcity first (fewest available get first pick
    # of their own quota, doesn't matter much here but keeps it deterministic)
    parents_sorted = sorted(parent_to_entities.keys(), key=lambda p: availability[p])

    for p in parents_sorted:
        n_pick = allocation[p]
        candidates = parent_to_entities[p]
        picked = balanced_pick_within_parent(candidates, entities, n_pick, used)
        for iri in picked:
            used.add(iri)
        selected.extend(picked)

    # If rounding left us short of TARGET_TOTAL, top up randomly from unused
    if len(selected) < TARGET_TOTAL:
        shortfall = TARGET_TOTAL - len(selected)
        remaining_pool = [iri for iri in entities if iri not in used]
        random.shuffle(remaining_pool)
        topup = remaining_pool[:shortfall]
        selected.extend(topup)
        used.update(topup)

    # If somehow over (shouldn't happen), trim randomly
    if len(selected) > TARGET_TOTAL:
        random.shuffle(selected)
        selected = selected[:TARGET_TOTAL]

    # Build output
    output = {iri: data[iri] for iri in selected}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Report ---
    print(f"\nSelected {len(selected)} entities -> {OUTPUT_PATH}\n")

    depth_counts = defaultdict(int)
    length_counts = defaultdict(int)
    parent_counts = defaultdict(int)

    for iri in selected:
        rec = entities[iri]
        depth_counts[rec.get("depth")] += 1
        length_counts[rec["length_bucket"]] += 1
        parents = rec.get("direct_parents", [])
        if not parents:
            parent_counts["__NO_PARENT__"] += 1
        else:
            for p in parents:
                parent_counts[p["local_name"]] += 1

    print("By depth:")
    for d, c in sorted(depth_counts.items(), key=lambda x: (x[0] is None, x[0])):
        print(f"  depth {d}: {c}")

    print("\nBy length bucket:")
    for lb in ["short", "medium", "long"]:
        print(f"  {lb}: {length_counts[lb]}")

    print(f"\nBy parent class (top 20 of {len(parent_counts)}):")
    for p, c in sorted(parent_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {p}: {c}")

    print(f"\nDistinct parent classes represented: {len(parent_counts)} / {len(parent_to_entities)}")


if __name__ == "__main__":
    main()