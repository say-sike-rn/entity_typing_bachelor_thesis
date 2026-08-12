#!/usr/bin/env python3
"""
Fix entity_registry.json: add per-class depth to direct_parents + actual_depth to entities
=============================================================================================
For every unique class IRI that appears in some entity's "direct_parents" list, this script
computes the SHORTEST PATH LENGTH (in rdfs:subClassOf hops) up to schema:Person, using the
same local QLever endpoint your pipeline already talks to.

Because SPARQL property paths (rdfs:subClassOf*) don't return hop-counts, we do this with
a level-by-level BFS *in SPARQL*:

  depth 0       = schema:Person itself
  depth 1       = { ?c : ?c rdfs:subClassOf schema:Person }
  depth 2       = { ?c : ?c rdfs:subClassOf <depth-1 class> } (excluding classes already
                    placed at a shallower depth)
  ... and so on, until either every target class has been placed, or a full BFS round finds
  no new classes (in which case whatever remains is unreachable -> depth -1).

Each BFS round is ONE SPARQL query (VALUES-bound frontier), so this scales fine even for a
large number of distinct parent classes.

Usage:
    python fix_entity_registry.py entity_registry.json \
        [--endpoint http://localhost:9004] \
        [--person-iri http://schema.org/Person] \
        [--out entity_registry_fixed.json] \
        [--max-depth 50] \
        [--cache depth_cache.json]

Notes:
    - Caches computed depths in --cache so re-runs (e.g. after adding more entities) don't
      re-query classes we already resolved.
    - Classes with no path to schema:Person get depth -1.
    - An entity's actual_depth = min(depth of its direct_parents); if ALL direct_parents are
      unreachable (-1) or the entity has no direct_parents at all, actual_depth is also -1
      and the entity IRI is printed in the final "flagged" summary so you can eyeball them.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import requests


def sparql_post(endpoint: str, query: str, timeout: int = 300) -> Dict:
    response = requests.post(
        endpoint,
        data=query.encode("utf-8"),
        headers={
            "Content-Type": "application/sparql-query",
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def fetch_direct_subclasses_of_batch(endpoint: str, parent_iris: List[str]) -> Dict[str, Set[str]]:
    """
    Given a batch of parent IRIs (the current BFS frontier), find all classes ?c such that
    ?c rdfs:subClassOf <parent> for some parent in the batch.

    Returns: parent_iri -> set of direct child IRIs found in the graph (unfiltered — caller
    intersects with the set of classes it's still looking for).
    """
    BATCH_SIZE = 300
    result: Dict[str, Set[str]] = {p: set() for p in parent_iris}

    for i in range(0, len(parent_iris), BATCH_SIZE):
        batch = parent_iris[i:i + BATCH_SIZE]
        values_block = " ".join(f"<{iri}>" for iri in batch)
        query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?child ?parent WHERE {{
    VALUES ?parent {{ {values_block} }}
    ?child rdfs:subClassOf ?parent .
}}
"""
        try:
            bindings = sparql_post(endpoint, query).get("results", {}).get("bindings", [])
            for b in bindings:
                child_iri = b["child"]["value"]
                parent_iri = b["parent"]["value"]
                if parent_iri in result:
                    result[parent_iri].add(child_iri)
        except Exception as exc:
            print(f"  [WARN] batch query failed for {len(batch)} parents: {exc}", file=sys.stderr)

    return result


def compute_depths_bfs(
    endpoint: str,
    targets: Set[str],
    person_iri: str,
    max_depth: int,
    cache: Dict[str, int],
) -> Dict[str, int]:
    """
    BFS outward from person_iri (depth 0) through rdfs:subClassOf children, level by level,
    until all `targets` are found or max_depth is hit. Returns {iri: depth} for every target
    (unresolved targets get -1). Uses/updates `cache` in place.
    """
    remaining = {t for t in targets if t not in cache}
    if not remaining:
        print(f"  All {len(targets)} target classes already in cache — nothing to query.")
        return {t: cache[t] for t in targets}

    print(f"  {len(targets) - len(remaining)} already cached, {len(remaining)} to resolve via BFS.")

    found: Dict[str, int] = {}
    frontier: Set[str] = {person_iri}
    visited: Set[str] = {person_iri}
    depth = 0

    while remaining and depth < max_depth and frontier:
        depth += 1
        print(f"  BFS depth {depth}: expanding frontier of {len(frontier)} class(es) ...")
        children_map = fetch_direct_subclasses_of_batch(endpoint, list(frontier))

        next_frontier: Set[str] = set()
        for parent, children in children_map.items():
            for child in children:
                if child in visited:
                    continue
                visited.add(child)
                next_frontier.add(child)
                if child in remaining:
                    found[child] = depth
                    cache[child] = depth
                    remaining.discard(child)

        print(f"    -> found {len(next_frontier)} new classes this round; "
              f"{len(remaining)} target(s) still unresolved.")
        frontier = next_frontier

    if remaining:
        print(f"  [INFO] {len(remaining)} class(es) unreachable from {person_iri} "
              f"within max_depth={max_depth} -> depth = -1")
        for r in remaining:
            found[r] = -1
            cache[r] = -1

    return {t: cache[t] for t in targets}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("registry_path", type=Path, help="Path to entity_registry.json")
    ap.add_argument("--endpoint", default="http://localhost:9004", help="QLever SPARQL endpoint")
    ap.add_argument("--person-iri", default="http://schema.org/Person",
                     help="Root class IRI to measure depth from (default: schema:Person)")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output path (default: overwrite input in place, atomically)")
    ap.add_argument("--max-depth", type=int, default=50, help="BFS safety limit")
    ap.add_argument("--cache", type=Path, default=Path("./class_depth_cache.json"),
                     help="Path to persist computed class depths across runs")
    args = ap.parse_args()

    registry_path: Path = args.registry_path
    out_path: Path = args.out or registry_path

    print(f"Loading {registry_path} ...")
    registry: Dict = json.loads(registry_path.read_text(encoding="utf-8"))
    print(f"  {len(registry)} entities loaded.")

    # Collect every unique direct_parents IRI across the whole registry
    target_classes: Set[str] = set()
    entities_missing_parents: List[str] = []
    for entity_iri, entity in registry.items():
        parents = entity.get("direct_parents", [])
        if not parents:
            entities_missing_parents.append(entity_iri)
            continue
        for p in parents:
            target_classes.add(p["iri"])

    print(f"Found {len(target_classes)} unique direct-parent classes across the registry.")
    if entities_missing_parents:
        print(f"  [NOTE] {len(entities_missing_parents)} entities have NO direct_parents at all.")

    # Load depth cache
    cache: Dict[str, int] = {}
    if args.cache.exists():
        cache = json.loads(args.cache.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached class depths from {args.cache}")

    print(f"\nComputing depths via BFS from {args.person_iri} ...")
    t0 = time.time()
    depths = compute_depths_bfs(
        endpoint=args.endpoint,
        targets=target_classes,
        person_iri=args.person_iri,
        max_depth=args.max_depth,
        cache=cache,
    )
    print(f"BFS complete in {time.time() - t0:.1f}s")

    # Persist cache
    args.cache.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Cache saved -> {args.cache} ({len(cache)} entries)")

    # Now write depth into each entity's direct_parents + actual_depth on the entity
    unreachable_classes = {c for c, d in depths.items() if d == -1}
    flagged_entities: List[str] = []

    for entity_iri, entity in registry.items():
        parents = entity.get("direct_parents", [])
        if not parents:
            entity["actual_depth"] = -1
            flagged_entities.append(entity_iri)
            continue

        parent_depths = []
        for p in parents:
            d = depths.get(p["iri"], -1)
            p["depth"] = d
            parent_depths.append(d)

        valid_depths = [d for d in parent_depths if d != -1]
        if valid_depths:
            entity["actual_depth"] = min(valid_depths)
        else:
            entity["actual_depth"] = -1
            flagged_entities.append(entity_iri)

    # Write atomically
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)
    print(f"\nFixed registry written -> {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Entities total                : {len(registry)}")
    print(f"Unique parent classes resolved: {len(target_classes)}")
    print(f"Unreachable classes (depth -1): {len(unreachable_classes)}")
    if unreachable_classes:
        print("  Sample unreachable classes:")
        for c in list(unreachable_classes)[:10]:
            print(f"    - {c}")
    print(f"Entities flagged (actual_depth = -1): {len(flagged_entities)}")
    if flagged_entities:
        print("  Sample flagged entities:")
        for e in flagged_entities[:10]:
            print(f"    - {e}")


if __name__ == "__main__":
    main()