#!/usr/bin/env python3
"""
YAGO → BERT Training Data Pipeline
====================================
Runs the full pipeline in one script:
  1. SPARQL query  → subclass CSV
     - First run  : fetches layer 1+2 (direct + grandchildren of schema:Person)
     - Later runs : fetches classes whose direct parent is already in subclass_registry.json
  1b. Filter subclasses already registered in ./subclass_registry.json
  2. Batch entity extraction → all_results.json
     - Entities already in entity_registry.json are skipped (deduplication)
  3. Filter cryptic _Q### entity names
  4. Cap total entities (keep richest subclasses)
  5. Restructure occupation-centric → entity-centric
  6. Fetch Wikipedia summaries (parallelized, ramp-up rate limiting)
  7. Validate remaining subclass labels
  8. Export {text, label} BERT training data
  9. Register newly trained subclasses in ./subclass_registry.json
     Register entities in ./entity_registry.json

Usage:
    python yago_pipeline.py [--endpoint URL] [--min-entities N]
                            [--limit N] [--min-desc N]
                            [--delay F] [--start-step N]
                            [--wiki-workers N]

Options:
    --endpoint      QLever endpoint        (default: http://localhost:9004)
    --min-entities  Min entities/class     (default: 200)
    --limit         Max entities/class     (default: 500)
    --min-desc      Min Wikipedia chars    (default: 200)
    --delay         Seconds between SPARQL queries (default: 0.0)
    --start-step    Resume from step N (1-8) (default: 1)
    --run-dir       Reuse existing run dir (required with --start-step > 1)
    --registry      Path to subclass registry (default: ./subclass_registry.json)
    --entity-registry Path to entity registry (default: ./entity_registry.json)
    --wiki-workers  Parallel Wikipedia workers (default: 3, burst mode — retries on 429)
    --max-total     Max total entities before Wikipedia step (default: 20000)
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import wikipediaapi
except ImportError:
    wikipediaapi = None


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

class RunLogger:
    """Writes to stdout AND a persistent log file simultaneously."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self._f = open(log_path, "w", encoding="utf-8")
        self._lock = Lock()

    def log(self, msg: str = "", level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {msg}"
        with self._lock:
            print(line)
            self._f.write(line + "\n")
            self._f.flush()

    def separator(self, char: str = "─", width: int = 72):
        line = char * width
        with self._lock:
            print(line)
            self._f.write(line + "\n")
            self._f.flush()

    def section(self, title: str):
        self.separator("═")
        self.log(f"  {title}")
        self.separator("═")

    def close(self):
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# Subclass registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_registry(registry_path: Path) -> Dict[str, Any]:
    """
    Load the subclass registry from disk.

    Schema:
    {
      "http://schema.org/Politician": {
        "local_name": "Politician",
        "registered_at": "2025-01-15T10:23:00",
        "run_dir": "yago_run_20250115_102300"
      },
      ...
    }
    """
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return {}


def save_registry(registry_path: Path, registry: Dict[str, Any]) -> None:
    """Atomically write the registry (write to .tmp, then rename)."""
    tmp = registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(registry_path)


def register_subclasses(
    registry_path: Path,
    registry: Dict[str, Any],
    new_subclasses: List[Dict],
    run_dir: str,
    log: "RunLogger",
) -> None:
    log.section("REGISTRY — Registering completed subclasses")
    now = datetime.now().isoformat()
    added = already = 0

    for sc in new_subclasses:
        iri = sc["iri"]
        if iri in registry:
            already += 1
            log.log(f"  (already registered, skipping)  {sc['name']}", "WARN")
        else:
            registry[iri] = {
                "local_name": sc["name"],
                "registered_at": now,
                "run_dir": run_dir,
            }
            log.log(f"  ✓ registered  {sc['name']}")
            added += 1

    save_registry(registry_path, registry)
    log.log(f"\nAdded to registry : {added}")
    log.log(f"Already present   : {already}")
    log.log(f"Registry total    : {len(registry)}")
    log.log(f"Registry path     : {registry_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Entity registry helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_entity_registry(entity_registry_path: Path) -> Dict[str, Any]:
    """
    Load the entity registry from disk.

    Schema:
    {
      "http://yago-knowledge.org/resource/Angela_Merkel": {
        "local_name": "Angela_Merkel",
        "source_class": "Politician",
        "source_class_iri": "http://schema.org/Politician",
        "depth": 1,
        "registered_at": "2025-01-15T10:23:00",
        "run_dir": "yago_run_20250115_102300"
      },
      ...
    }

    Note: depth 1 = layer 1+2 merged run, depth 2 = first child-layer run, etc.
    """
    if entity_registry_path.exists():
        return json.loads(entity_registry_path.read_text(encoding="utf-8"))
    return {}


def save_entity_registry(entity_registry_path: Path, entity_registry: Dict[str, Any]) -> None:
    """Atomically write the entity registry."""
    tmp = entity_registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entity_registry, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(entity_registry_path)


def register_entities(
    entity_registry_path: Path,
    entity_registry: Dict[str, Any],
    new_entities: List[Dict],  # list of {iri, local_name, source_class, source_class_iri, depth}
    run_dir: str,
    log: "RunLogger",
) -> None:
    """Append newly trained entities to the on-disk entity registry."""
    log.section("ENTITY REGISTRY — Registering entities")
    now = datetime.now().isoformat()
    added = already = 0

    for ent in new_entities:
        iri = ent["iri"]
        if iri in entity_registry:
            already += 1
        else:
            entity_registry[iri] = {
                "local_name": ent["local_name"],
                "source_class": ent["source_class"],
                "source_class_iri": ent["source_class_iri"],
                "depth": ent["depth"],
                "registered_at": now,
                "run_dir": run_dir,
                "description": ent.get("description", ""),
                "direct_parents": ent.get("direct_parents", []),
            }
            added += 1

    save_entity_registry(entity_registry_path, entity_registry)
    log.log(f"Entities added to registry : {added}")
    log.log(f"Entities already present   : {already}")
    log.log(f"Entity registry total      : {len(entity_registry)}")
    log.log(f"Entity registry path       : {entity_registry_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# SPARQL helpers
# ─────────────────────────────────────────────────────────────────────────────

# Used on the first run:
SUBCLASS_QUERY_BOOTSTRAP = """
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX yago:   <http://yago-knowledge.org/resource/>

SELECT ?subclass (COUNT(DISTINCT ?instance) AS ?instanceCount)
WHERE {
    {
        ?subclass rdfs:subClassOf schema:Person .
        ?instance a ?subclass .
    }
}
GROUP BY ?subclass
HAVING (COUNT(DISTINCT ?instance) >= 250)
"""


def build_subclass_query_from_registry(registry: Dict[str, Any]) -> str:
    """
    For runs after the first: fetch only classes whose rdfs:subClassOf is
    an IRI already present in the subclass registry.

    Generates a UNION of one VALUES block per registered parent, asking
    for direct children of each.
    """
    registered_iris = list(registry.keys())

    # Build a VALUES clause listing all registered IRIs as potential parents.
    values_block = " ".join(f"<{iri}>" for iri in registered_iris)

    return f"""
        PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX schema: <http://schema.org/>
        PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?subclass (COUNT(DISTINCT ?instance) AS ?instanceCount)
        WHERE {{
            VALUES ?parent {{ {values_block} }}
            ?subclass rdfs:subClassOf ?parent .
            ?instance a ?subclass .
        }}
        GROUP BY ?subclass
        HAVING (COUNT(DISTINCT ?instance) >= 250)
    """


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – Run SPARQL query → subclass CSV
# ─────────────────────────────────────────────────────────────────────────────

def step1_query_subclasses(run: "PipelineRun") -> Path:
    log = run.log
    log.section("STEP 1 — SPARQL query: subclasses of schema:Person")

    is_bootstrap = len(run.registry) == 0
    if is_bootstrap:
        log.log("Registry is empty → bootstrap run: fetching layer 1+2 (merged).")
        query = SUBCLASS_QUERY_BOOTSTRAP
    else:
        log.log(f"Registry has {len(run.registry)} entries → fetching direct children of registered classes.")
        query = build_subclass_query_from_registry(run.registry)

    query_file = run.dir / "query.sparql"
    query_file.write_text(query, encoding="utf-8")
    log.log(f"Query saved → {query_file}")

    log.log("Sending query to endpoint …")
    results = sparql_post(run.cfg["endpoint"], query)
    bindings = results.get("results", {}).get("bindings", [])
    log.log(f"Raw result rows: {len(bindings)}")

    csv_path = run.dir / "subclasses.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subclass", "instanceCount"])
        writer.writeheader()
        for row in bindings:
            writer.writerow({
                "subclass": row["subclass"]["value"],
                "instanceCount": row["instanceCount"]["value"],
            })

    log.log(f"Subclasses CSV saved → {csv_path}  ({len(bindings)} rows)")
    run.meta["step1_rows"] = len(bindings)
    run.meta["step1_bootstrap"] = is_bootstrap
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b – Filter out already-registered subclasses
# ─────────────────────────────────────────────────────────────────────────────

def step1b_filter_registry(run: "PipelineRun", csv_path: Path) -> Path:
    """
    Remove subclasses already present in the subclass registry.
    On bootstrap runs this is a no-op (registry is empty).
    """
    log = run.log
    log.section("STEP 1b — Filter already-registered subclasses")

    registry = run.registry
    log.log(f"Registry contains {len(registry)} previously trained subclasses.")

    rows: List[Dict] = []
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    kept: List[Dict] = []
    skipped: List[str] = []

    for row in rows:
        iri = row["subclass"].strip()
        if iri in registry:
            local = registry[iri]["local_name"]
            registered_in = registry[iri].get("run_dir", "unknown run")
            log.log(f"  ✗ SKIP (already trained in {registered_in})  {local}")
            skipped.append(iri)
        else:
            kept.append(row)

    log.log(f"\nSubclasses from query    : {len(rows)}")
    log.log(f"Already in registry      : {len(skipped)}")
    log.log(f"Proceeding with          : {len(kept)}")

    if not kept:
        log.log(
            "\n⚠  All subclasses from this query are already registered. "
            "Nothing to do — exiting early.",
            "WARN",
        )
        run.save_meta()
        log.separator("═")
        log.log("  Pipeline exited early: no new subclasses.")
        log.separator("═")
        log.close()
        sys.exit(0)

    filtered_csv = run.dir / "subclasses_filtered.csv"
    with open(filtered_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subclass", "instanceCount"])
        writer.writeheader()
        writer.writerows(kept)

    log.log(f"Filtered CSV → {filtered_csv}")

    run.new_subclasses = [
        {"iri": row["subclass"].strip(), "name": _local_name(row["subclass"].strip())}
        for row in kept
    ]

    run.meta["step1b_total_from_query"] = len(rows)
    run.meta["step1b_skipped_registered"] = len(skipped)
    run.meta["step1b_proceeding"] = len(kept)

    return filtered_csv


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – Batch entity extraction → all_results.json
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_QUERY_TMPL = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?entity ?entityType WHERE {{
    ?entity rdf:type <{class_iri}> .
    BIND(<{class_iri}> AS ?entityType)
    FILTER(?entity != <{class_iri}>)
}}
ORDER BY ?entity
LIMIT {limit}
"""


def _local_name(iri: str) -> str:
    return iri.split("/")[-1].split("#")[-1]


def _fetch_entities(endpoint: str, class_iri: str, limit: int, delay: float) -> List[Dict]:
    query = ENTITY_QUERY_TMPL.format(class_iri=class_iri, limit=limit)
    time.sleep(delay)
    try:
        results = sparql_post(endpoint, query)
        bindings = results.get("results", {}).get("bindings", [])
        return [
            {
                "entity_iri": b["entity"]["value"],
                "entity_local_name": _local_name(b["entity"]["value"]),
                "type_iri": b["entityType"]["value"],
                "type_local_name": _local_name(b["entityType"]["value"]),
            }
            for b in bindings
        ]
    except Exception:
        return []


def _fetch_direct_parents_batch(endpoint: str, entity_iris: List[str]) -> Dict[str, List[Dict]]:
    """
    For a batch of entity IRIs, fetch all direct rdf:type values.
    Returns a dict: entity_iri → [{"iri": ..., "local_name": ...}, ...]

    Uses a VALUES clause so the whole batch is one SPARQL round-trip.
    Batches of 500 keep query size manageable for QLever.
    """
    BATCH_SIZE = 500
    result: Dict[str, List[Dict]] = {iri: [] for iri in entity_iris}

    for i in range(0, len(entity_iris), BATCH_SIZE):
        batch = entity_iris[i:i + BATCH_SIZE]
        values_block = " ".join(f"<{iri}>" for iri in batch)
        query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?entity ?parent WHERE {{
    VALUES ?entity {{ {values_block} }}
    ?entity rdf:type ?parent .
}}
"""
        try:
            bindings = sparql_post(endpoint, query).get("results", {}).get("bindings", [])
            for b in bindings:
                ent_iri = b["entity"]["value"]
                par_iri = b["parent"]["value"]
                if ent_iri in result:
                    result[ent_iri].append({
                        "iri": par_iri,
                        "local_name": _local_name(par_iri),
                    })
        except Exception:
            pass  # batch failed; entities keep empty lists, logged in step2

    return result


def step2_batch_extract(run: "PipelineRun", csv_path: Path) -> Path:
    log = run.log
    cfg = run.cfg
    log.section("STEP 2 — Batch entity extraction")

    subclasses: List[Dict] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iri = row["subclass"].strip()
            if iri:
                subclasses.append({"iri": iri, "name": _local_name(iri)})

    log.log(f"Classes to process: {len(subclasses)}")
    log.log(f"Limit per class: {cfg['limit']}  |  Min entities: {cfg['min_entities']}")

    # ── Entity deduplication ────────────────────────────────────────────────
    # Entities already in the entity registry are skipped so they don't appear
    # in multiple training runs. Each entity is assigned to exactly one class
    # (the first one that fetched it).
    #
    # To DISABLE entity deduplication, comment out the block below (lines
    # between "DEDUP ON" and "DEDUP OFF") and uncomment the "DEDUP OFF" line.
    #
    # DEDUP ON ▼▼▼
    entity_registry = run.entity_registry
    seen_this_run: set = set()   # tracks IRIs added in *this* run
    dedup_enabled = True
    # DEDUP OFF (uncomment to disable):
    # entity_registry = {}
    # seen_this_run = set()
    # dedup_enabled = False
    # ▲▲▲ DEDUP ON

    all_results: Dict[str, Any] = {}
    skipped_classes: List[str] = []
    total_dedup_skipped = 0

    for idx, sc in enumerate(subclasses, 1):
        raw_entities = _fetch_entities(
            cfg["endpoint"], sc["iri"], cfg["limit"], cfg["delay"]
        )

        if dedup_enabled:
            filtered_entities = []
            for e in raw_entities:
                iri = e["entity_iri"]
                if iri in entity_registry or iri in seen_this_run:
                    total_dedup_skipped += 1
                else:
                    seen_this_run.add(iri)
                    filtered_entities.append(e)
            entities = filtered_entities
        else:
            entities = raw_entities

        count = len(entities)
        status = "✓" if count >= cfg["min_entities"] else "✗"
        dedup_note = f"  (dedup skipped {len(raw_entities) - count})" if dedup_enabled else ""
        log.log(f"  [{idx:>3}/{len(subclasses)}] {status} {sc['name']:40s}  {count} entities{dedup_note}")

        if count >= cfg["min_entities"]:
            all_results[sc["name"]] = {
                "iri": sc["iri"],
                "entities": entities,
                "count": count,
                "query_time": datetime.now().isoformat(),
                "meets_threshold": True,
            }
        else:
            skipped_classes.append(sc["name"])

    # ── Fetch direct parents for all kept entities in batches ─────────────
    all_entity_iris = [
        e["entity_iri"]
        for cat in all_results.values()
        for e in cat["entities"]
    ]
    log.log(f"\nFetching direct parents for {len(all_entity_iris)} entities in batches of 500 …")
    parents_map = _fetch_direct_parents_batch(cfg["endpoint"], all_entity_iris)

    no_parents_count = 0
    for cat in all_results.values():
        for e in cat["entities"]:
            parents = parents_map.get(e["entity_iri"], [])
            e["direct_parents"] = parents
            if not parents:
                no_parents_count += 1

    if no_parents_count:
        log.log(f"  ⚠  {no_parents_count} entities returned zero parents (check YAGO typing)", "WARN")

    out = run.dir / "step2_all_results.json"
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    log.log(f"\nKept classes  : {len(all_results)}")
    log.log(f"Skipped (< {cfg['min_entities']} entities): {len(skipped_classes)}")
    if dedup_enabled:
        log.log(f"Entities skipped by deduplication: {total_dedup_skipped}")
    if skipped_classes:
        log.log(f"  → {', '.join(skipped_classes[:10])}{'…' if len(skipped_classes) > 10 else ''}")
    log.log(f"Output → {out}")

    # Trim registry candidates to classes that passed the threshold
    passed_iris = {v["iri"] for v in all_results.values()}
    run.new_subclasses = [sc for sc in run.new_subclasses if sc["iri"] in passed_iris]
    log.log(
        f"\n  (Registry: {len(run.new_subclasses)} subclasses eligible for registration "
        f"after entity-count filter)"
    )

    run.meta["step2_kept"] = len(all_results)
    run.meta["step2_skipped"] = len(skipped_classes)
    run.meta["step2_dedup_skipped"] = total_dedup_skipped
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – Filter _Q### cryptic names
# ─────────────────────────────────────────────────────────────────────────────

_CRYPTIC_RE = re.compile(r"_Q\d+$|^Q\d+$")


def step3_filter_q(run: "PipelineRun", in_path: Path) -> Path:
    log = run.log
    log.section("STEP 3 — Filter cryptic _Q### entity names")

    data = json.loads(in_path.read_text(encoding="utf-8"))

    before = after = 0
    for cat in data.values():
        orig = cat.get("entities", [])
        before += len(orig)
        cat["entities"] = [e for e in orig if not _CRYPTIC_RE.search(e["entity_local_name"])]
        after += len(cat["entities"])

    removed = before - after
    log.log(f"Entities before: {before}")
    log.log(f"Entities after : {after}  (removed {removed} cryptic names)")

    out = run.dir / "step3_no_q.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.log(f"Output → {out}")

    run.meta["step3_removed_q"] = removed
    run.meta["step3_entities"] = after
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 – Cap total entities at MAX_TOTAL by keeping richest subclasses
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TOTAL = 20_000


def step4_cap_entities(run: "PipelineRun", in_path: Path) -> Path:
    log = run.log
    log.section("STEP 4 — Cap total entities (keep richest subclasses)")

    max_total = run.cfg.get("max_total", DEFAULT_MAX_TOTAL)
    data = json.loads(in_path.read_text(encoding="utf-8"))

    class_counts = [
        (name, len(cat.get("entities", [])))
        for name, cat in data.items()
    ]
    total_before = sum(c for _, c in class_counts)

    log.log(f"Total entities before cap : {total_before}")
    log.log(f"Cap threshold             : {max_total}")

    if total_before <= max_total:
        log.log("✓ Under the cap — nothing to trim.")
        out = run.dir / "step4_capped.json"
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        run.meta["step4_capped"] = False
        run.meta["step4_classes_kept"] = len(class_counts)
        run.meta["step4_entities_after"] = total_before
        log.log(f"Output → {out}")
        return out

    class_counts_sorted = sorted(class_counts, key=lambda x: -x[1])
    kept_names: List[str] = []
    running_total = 0
    dropped_names: List[str] = []

    for name, count in class_counts_sorted:
        if running_total + count <= max_total:
            kept_names.append(name)
            running_total += count
        else:
            dropped_names.append(name)

    log.log(f"\nClasses kept    : {len(kept_names)}  ({running_total} entities)")
    log.log(f"Classes dropped : {len(dropped_names)}")
    log.log("\nKept (richest first):")
    for name in kept_names:
        cnt = dict(class_counts)[name]
        log.log(f"  ✓  {name:40s}  {cnt:>5}")
    log.log("\nDropped (would have exceeded cap):")
    for name in dropped_names:
        cnt = dict(class_counts)[name]
        log.log(f"  ✗  {name:40s}  {cnt:>5}")

    trimmed = {name: data[name] for name in kept_names}
    out = run.dir / "step4_capped.json"
    out.write_text(json.dumps(trimmed, indent=2, ensure_ascii=False), encoding="utf-8")

    kept_iris = {data[name]["iri"] for name in kept_names}
    before_reg = len(run.new_subclasses)
    run.new_subclasses = [sc for sc in run.new_subclasses if sc["iri"] in kept_iris]
    dropped_reg = before_reg - len(run.new_subclasses)
    if dropped_reg:
        log.log(
            f"\n  (Registry: removed {dropped_reg} cap-dropped subclasses from "
            f"registration candidates; {len(run.new_subclasses)} remain)"
        )

    run.meta["step4_capped"] = True
    run.meta["step4_classes_kept"] = len(kept_names)
    run.meta["step4_classes_dropped"] = len(dropped_names)
    run.meta["step4_entities_before"] = total_before
    run.meta["step4_entities_after"] = running_total
    log.log(f"\nOutput → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 – Restructure: occupation-centric → entity-centric
# ─────────────────────────────────────────────────────────────────────────────

def step5_restructure(run: "PipelineRun", in_path: Path) -> Path:
    log = run.log
    log.section("STEP 5 — Restructure: occupation-centric → entity-centric")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    entities: Dict[str, Any] = {}

    for occ_name, occ_data in data.items():
        for e in occ_data.get("entities", []):
            iri = e["entity_iri"]
            if iri not in entities:
                entities[iri] = {
                    "entity_iri": iri,
                    "entity_local_name": e["entity_local_name"],
                    "source_class": occ_name,
                    "source_class_iri": occ_data["iri"],
                    "types": [],
                    "description": "",
                    "direct_parents": e.get("direct_parents", []),
                }
            entities[iri]["types"].append({
                "type_iri": e.get("type_iri", occ_data["iri"]),
                "type_local_name": e.get("type_local_name", occ_name),
                "source_occupation": occ_name,
            })

    out_data = {"entities": list(entities.values())}
    out = run.dir / "step5_restructured.json"
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")

    log.log(f"Unique entities after dedup: {len(entities)}")
    log.log(f"Output → {out}")

    run.meta["step5_unique_entities"] = len(entities)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 – Fetch Wikipedia summaries (parallelized with ramp-up)
# ─────────────────────────────────────────────────────────────────────────────

def decode_yago_title(title: str) -> str:
    return re.sub(
        r'_u([0-9A-Fa-f]{4})_',
        lambda m: chr(int(m.group(1), 16)),
        title
    )


_WIKI_MAX_RETRIES = 3
_WIKI_BACKOFF_BASE = 2.0  # seconds; doubles on each retry

_WIKI_API_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
_WIKI_HEADERS = {"User-Agent": "YAGOPipeline/1.0 (karen.awe@mytum.de)"}


def _fetch_one_wiki(entity: Dict, min_desc: int) -> Tuple[Optional[Dict], str]:
    """
    Fetch the Wikipedia summary for a single entity via the REST API.

    Uses the /page/summary endpoint which is designed for programmatic
    access and is much faster than the HTML API used by wikipediaapi.

    Returns (enriched_entity_or_None, status)
    status: "ok" | "no_page" | "too_short" | "error"
    """
    name = entity["entity_local_name"]
    wiki_title = decode_yago_title(name)
    url = _WIKI_API_URL.format(title=wiki_title)
    
    for attempt in range(_WIKI_MAX_RETRIES):
        time.sleep(0.2)
        try:
            response = requests.get(url, headers=_WIKI_HEADERS, timeout=10)

            if response.status_code == 404:
                return None, "no_page"

            if response.status_code == 429:
                backoff = _WIKI_BACKOFF_BASE * (5 ** attempt)
                print(f"[429] rate limited on {name}, backing off {backoff:.0f}s …")
                time.sleep(backoff)
                continue

            if response.status_code != 200:
                if attempt < _WIKI_MAX_RETRIES - 1:
                    time.sleep(_WIKI_BACKOFF_BASE)
                    continue
                return None, "error"

            summary = response.json().get("extract", "")
            
            print(time.ctime(), " Fetched summary for", name, "length:", len(summary))
            if not summary:
                return None, "no_page"

        except Exception as exc:
            if attempt < _WIKI_MAX_RETRIES - 1:
                time.sleep(_WIKI_BACKOFF_BASE)
                continue
            return None, "error"

        if len(summary) < min_desc:
            return None, "too_short"

        entity = dict(entity)
        entity["description"] = summary[:60_000]
        return entity, "ok"

    return None, "error"

def step6_wikipedia(run: "PipelineRun", in_path: Path) -> Path:
    log = run.log
    log.section("STEP 6 — Fetch Wikipedia summaries (parallelized, burst)")

    if wikipediaapi is None:
        log.log("ERROR: 'wikipedia-api' package not installed. Run: pip install wikipedia-api", "ERROR")
        sys.exit(1)

    min_desc = run.cfg["min_desc"]
    max_workers = run.cfg.get("wiki_workers", 3)

    log.log(f"Workers        : {max_workers}  (burst — retries on 429 only)")
    log.log(f"Min desc chars : {min_desc}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    entities_in = data["entities"]
    total = len(entities_in)

    kept: List[Dict] = []
    stats = {"no_page": 0, "too_short": 0, "ok": 0, "error": 0}
    stats_lock = Lock()
    kept_lock = Lock()

    log.log(f"Fetching descriptions for {total} entities with {max_workers} worker(s) …")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one_wiki, entity, min_desc): entity
            for entity in entities_in
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result, status = future.result()

            with stats_lock:
                stats[status] = stats.get(status, 0) + 1

            if result is not None:
                with kept_lock:
                    kept.append(result)

            # Progress log every 100 completions
            if completed % 100 == 0 or completed == total:
                with stats_lock:
                    log.log(
                        f"  Progress: {completed}/{total}  "
                        f"ok={stats['ok']}  no_page={stats['no_page']}  "
                        f"too_short={stats['too_short']}  error={stats['error']}"
                    )

            # Checkpoint every 500 successful fetches
            with stats_lock:
                ok_count = stats["ok"]
            if ok_count > 0 and ok_count % 500 == 0:
                ckpt = run.dir / f"checkpoint_step6_{ok_count}.json"
                with kept_lock:
                    ckpt_data = {"entities": list(kept)}
                ckpt.write_text(json.dumps(ckpt_data, indent=2, ensure_ascii=False), encoding="utf-8")
                log.log(f"  Checkpoint → {ckpt}")

    data["entities"] = kept
    out = run.dir / "step6_with_descriptions.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    log.log(f"\nKept        : {stats['ok']}")
    log.log(f"No page     : {stats['no_page']}")
    log.log(f"Too short   : {stats['too_short']}")
    log.log(f"Errors      : {stats['error']}")
    log.log(f"Output → {out}")

    run.meta.update({f"step6_{k}": v for k, v in stats.items()})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 – Validate that remaining subclass labels still have enough entities
# ─────────────────────────────────────────────────────────────────────────────

def step7_validate_labels(run: "PipelineRun", in_path: Path) -> Path:
    log = run.log
    log.section("STEP 7 — Validate and enforce subclass label minimums")

    MIN_LABEL_ENTITIES = 150

    data = json.loads(in_path.read_text(encoding="utf-8"))
    entities = data["entities"]

    label_counts: Dict[str, int] = {}
    for e in entities:
        for t in e.get("types", []):
            occ = t["source_occupation"]
            label_counts[occ] = label_counts.get(occ, 0) + 1

    passing  = {lbl for lbl, cnt in label_counts.items() if cnt >= MIN_LABEL_ENTITIES}
    dropping = {lbl for lbl, cnt in label_counts.items() if cnt <  MIN_LABEL_ENTITIES}

    kept_entities: List[Dict] = []
    for e in entities:
        e["types"] = [t for t in e.get("types", []) if t["source_occupation"] in passing]
        if e["types"]:
            kept_entities.append(e)

    data["entities"] = kept_entities

    log.log(f"Min entities threshold : {MIN_LABEL_ENTITIES}")
    log.log(f"Labels before          : {len(label_counts)}")
    log.log(f"Labels dropped         : {len(dropping)}")
    log.log(f"Labels kept            : {len(passing)}")
    log.log(f"Entities before        : {len(entities)}")
    log.log(f"Entities after         : {len(kept_entities)}")

    log.log("\nKept labels:")
    for lbl in sorted(passing, key=lambda l: -label_counts[l]):
        log.log(f"  ✓  {lbl:40s}  {label_counts[lbl]:>5}")

    if dropping:
        log.log("\nDropped labels (below threshold):")
        for lbl in sorted(dropping, key=lambda l: -label_counts[l]):
            log.log(f"  ✗  {lbl:40s}  {label_counts[lbl]:>5}")

    report = run.dir / "step7_label_report.csv"
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "entity_count", "dropped"])
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            w.writerow([lbl, cnt, "YES" if lbl in dropping else ""])

    log.log(f"\nLabel report → {report}")

    before_reg = len(run.new_subclasses)
    run.new_subclasses = [sc for sc in run.new_subclasses if sc["name"] in passing]
    dropped_reg = before_reg - len(run.new_subclasses)
    if dropped_reg:
        log.log(
            f"(Registry: removed {dropped_reg} dropped subclasses from "
            f"registration candidates; {len(run.new_subclasses)} remain)"
        )

    run.meta["step7_labels_before"]   = len(label_counts)
    run.meta["step7_labels_kept"]     = len(passing)
    run.meta["step7_labels_dropped"]  = len(dropping)
    run.meta["step7_entities_before"] = len(entities)
    run.meta["step7_entities_after"]  = len(kept_entities)

    out = run.dir / "step7_validated.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 – Export BERT training data {text, label}
#          + collect entities for entity registry
# ─────────────────────────────────────────────────────────────────────────────

def step8_make_bert(run: "PipelineRun", in_path: Path) -> Path:
    log = run.log
    log.section("STEP 8 — Export BERT training data")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    training: List[Dict] = []

    # Determine current run depth:
    #   bootstrap (registry was empty at run start) → depth 1
    #   subsequent runs → depth of deepest registered parent + 1
    #   (simple heuristic: len(registry) > 0 means at least one layer is done)
    is_bootstrap = run.meta.get("step1_bootstrap", len(run.registry) == 0)
    run_depth = 1 if is_bootstrap else 2  # Not very elegant way of determining depth. I swear I changed it for each iteration accordingly.

    new_entity_records: List[Dict] = []

    for entity in data["entities"]:
        text = entity.get("description", "").strip()
        if not text:
            continue
        types = entity.get("types", [])
        if not types:
            continue
        # Label = source_class stored on the entity (set in step5, first occurrence wins)
        label = entity.get("source_class", types[0]["source_occupation"])
        training.append({"text": text, "label": label})

        # Collect entity record for entity registry
        new_entity_records.append({
            "iri": entity["entity_iri"],
            "local_name": entity["entity_local_name"],
            "source_class": label,
            "source_class_iri": entity.get("source_class_iri", ""),
            "depth": run_depth,
            "description": text[:60_000],
            "direct_parents": entity.get("direct_parents", []),
        })

    # Filter labels with too few examples
    MIN_LABEL_ENTITIES = 150
    label_dist: Dict[str, int] = {}
    for ex in training:
        label_dist[ex["label"]] = label_dist.get(ex["label"], 0) + 1

    before = len(training)
    training = [ex for ex in training if label_dist[ex["label"]] >= MIN_LABEL_ENTITIES]
    after = len(training)
    surviving_labels = {ex["label"] for ex in training}

    # Mirror the filter to entity records
    new_entity_records = [
        r for r in new_entity_records if r["source_class"] in surviving_labels
    ]

    removed_examples = before - after
    dropped_labels = [lbl for lbl, cnt in label_dist.items() if cnt < MIN_LABEL_ENTITIES]

    log.log(f"Removed {removed_examples} examples from labels with < {MIN_LABEL_ENTITIES} entities")
    if dropped_labels:
        log.log(f"Dropped labels: {dropped_labels}")

    out = run.dir / "training_data.json"
    out.write_text(json.dumps(training, indent=2, ensure_ascii=False), encoding="utf-8")

    log.log(f"Training examples: {len(training)}")
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        if lbl in surviving_labels:
            log.log(f"  {lbl:40s}  {cnt:>5}")
    log.log(f"Output → {out}")

    # Trim subclass registry candidates to surviving labels
    before_reg = len(run.new_subclasses)
    run.new_subclasses = [
        sc for sc in run.new_subclasses if sc["name"] in surviving_labels
    ]
    dropped_reg = before_reg - len(run.new_subclasses)
    if dropped_reg:
        log.log(
            f"\n  (Registry: removed {dropped_reg} subclasses that produced too few "
            f"training examples; {len(run.new_subclasses)} remain)"
        )

    run.meta["step8_examples"] = len(training)
    run.meta["step8_new_entity_records"] = len(new_entity_records)

    # Stash entity records for the final registry commit
    run.new_entity_records = new_entity_records

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRun:
    def __init__(self, cfg: Dict, run_dir: Optional[Path] = None):
        self.cfg = cfg
        if run_dir:
            self.dir = run_dir
            self.dir.mkdir(parents=True, exist_ok=True)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.dir = Path(f"yago_run_{ts}")
            self.dir.mkdir(parents=True, exist_ok=True)

        self.log = RunLogger(self.dir / "run.log")
        self.meta: Dict[str, Any] = {
            "run_dir": str(self.dir),
            "started_at": datetime.now().isoformat(),
            "config": cfg,
        }

        # Subclass registry
        self.registry_path: Path = cfg["registry_path"]
        self.registry: Dict[str, Any] = load_registry(self.registry_path)

        # Entity registry
        self.entity_registry_path: Path = cfg["entity_registry_path"]
        self.entity_registry: Dict[str, Any] = load_entity_registry(self.entity_registry_path)

        # Populated by step1b, pruned through steps 2/4/7/8
        self.new_subclasses: List[Dict] = []

        # Populated by step8, committed after step8
        self.new_entity_records: List[Dict] = []

    def save_meta(self):
        self.meta["finished_at"] = datetime.now().isoformat()
        meta_path = self.dir / "run_meta.json"

        def _json_default(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        meta_path.write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        self.log.log(f"\nRun metadata → {meta_path}")

    def _find_step_file(self, step) -> Optional[Path]:
        mapping = {
            1:    "subclasses.csv",
            "1b": "subclasses_filtered.csv",
            2:    "step2_all_results.json",
            3:    "step3_no_q.json",
            4:    "step4_capped.json",
            5:    "step5_restructured.json",
            6:    "step6_with_descriptions.json",
            7:    "step7_validated.json",
            8:    "training_data.json",
        }
        p = self.dir / mapping[step]
        return p if p.exists() else None


def run_pipeline(cfg: Dict, start_step: int = 1, run_dir: Optional[Path] = None):
    run = PipelineRun(cfg, run_dir=run_dir)
    log = run.log

    log.separator("═")
    log.log("  YAGO → BERT Training Data Pipeline")
    log.log(f"  Run directory      : {run.dir}")
    log.log(f"  Endpoint           : {cfg['endpoint']}")
    log.log(f"  Min entities       : {cfg['min_entities']}")
    log.log(f"  Limit/class        : {cfg['limit']}")
    log.log(f"  Min desc chars     : {cfg['min_desc']}")
    log.log(f"  Query delay        : {cfg['delay']}s")
    log.log(f"  Max total entities : {cfg.get('max_total', DEFAULT_MAX_TOTAL)}")
    log.log(f"  Wiki workers       : {cfg.get('wiki_workers', 3)}  (burst mode)")
    log.log(f"  Subclass registry  : {cfg['registry_path'].resolve()}  ({len(run.registry)} entries)")
    log.log(f"  Entity registry    : {cfg['entity_registry_path'].resolve()}  ({len(run.entity_registry)} entries)")
    log.separator("═")

    pipeline_succeeded = False

    try:
        # ── Step 1 ──────────────────────────────────────────────────────────
        if start_step <= 1:
            csv_path = step1_query_subclasses(run)
        else:
            csv_path = run._find_step_file(1)
            log.log(f"[STEP 1 SKIPPED] Using existing file: {csv_path}")

        # ── Step 1b ─────────────────────────────────────────────────────────
        if start_step <= 2:
            filtered_csv = step1b_filter_registry(run, csv_path)
        else:
            filtered_csv = run._find_step_file("1b")
            if filtered_csv and filtered_csv.exists():
                log.log(f"[STEP 1b SKIPPED] Using existing file: {filtered_csv}")
                with open(filtered_csv, encoding="utf-8") as f:
                    run.new_subclasses = [
                        {"iri": row["subclass"].strip(), "name": _local_name(row["subclass"].strip())}
                        for row in csv.DictReader(f)
                    ]
            else:
                log.log("[STEP 1b SKIPPED] No filtered CSV found — registry filtering skipped.", "WARN")
                filtered_csv = csv_path

        # ── Step 2 ──────────────────────────────────────────────────────────
        if start_step <= 2:
            s2_path = step2_batch_extract(run, filtered_csv)
        else:
            s2_path = run._find_step_file(2)
            log.log(f"[STEP 2 SKIPPED] Using existing file: {s2_path}")

        # ── Step 3 ──────────────────────────────────────────────────────────
        if start_step <= 3:
            s3_path = step3_filter_q(run, s2_path)
        else:
            s3_path = run._find_step_file(3)
            log.log(f"[STEP 3 SKIPPED] Using existing file: {s3_path}")

        # ── Step 4 — cap entities ────────────────────────────────────────────
        if start_step <= 4:
            s4_path = step4_cap_entities(run, s3_path)
        else:
            s4_path = run._find_step_file(4)
            log.log(f"[STEP 4 SKIPPED] Using existing file: {s4_path}")

        # ── Step 5 — restructure ─────────────────────────────────────────────
        if start_step <= 5:
            s5_path = step5_restructure(run, s4_path)
        else:
            s5_path = run._find_step_file(5)
            log.log(f"[STEP 5 SKIPPED] Using existing file: {s5_path}")

        # ── Step 6 — Wikipedia ───────────────────────────────────────────────
        if start_step <= 6:
            s6_path = step6_wikipedia(run, s5_path)
        else:
            s6_path = run._find_step_file(6)
            log.log(f"[STEP 6 SKIPPED] Using existing file: {s6_path}")

        # ── Step 7 ──────────────────────────────────────────────────────────
        if start_step <= 7:
            s7_path = step7_validate_labels(run, s6_path)
        else:
            s7_path = run._find_step_file(7)
            log.log(f"[STEP 7 SKIPPED] Using existing file: {s7_path}")

        # ── Step 8 ──────────────────────────────────────────────────────────
        step8_make_bert(run, s7_path)

        pipeline_succeeded = True

    except KeyboardInterrupt:
        log.log("\nInterrupted by user.", "WARN")
    except Exception as exc:
        log.log(f"\nFATAL ERROR: {exc}", "ERROR")
        import traceback
        log.log(traceback.format_exc(), "ERROR")
    finally:
        if pipeline_succeeded:
            # Commit subclass registry
            if run.new_subclasses:
                register_subclasses(
                    registry_path=run.registry_path,
                    registry=run.registry,
                    new_subclasses=run.new_subclasses,
                    run_dir=str(run.dir),
                    log=log,
                )
            else:
                log.log("(No new subclasses to register for this run.)")

            # Commit entity registry
            if run.new_entity_records:
                register_entities(
                    entity_registry_path=run.entity_registry_path,
                    entity_registry=run.entity_registry,
                    new_entities=run.new_entity_records,
                    run_dir=str(run.dir),
                    log=log,
                )
            else:
                log.log("(No new entities to register for this run.)")
        else:
            log.log(
                "⚠  Pipeline did not complete successfully — registries NOT updated.",
                "WARN",
            )

        run.save_meta()
        log.separator("═")
        log.log("  Pipeline complete.")
        log.separator("═")
        log.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="YAGO → BERT Training Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--endpoint",        default="http://localhost:9004")
    p.add_argument("--min-entities",    type=int,   default=200)
    p.add_argument("--limit",           type=int,   default=500)
    p.add_argument("--min-desc",        type=int,   default=200)
    p.add_argument("--max-total",       type=int,   default=DEFAULT_MAX_TOTAL)
    p.add_argument("--delay",           type=float, default=0.0)
    p.add_argument("--wiki-workers",    type=int,   default=3,
                   help="Parallel Wikipedia workers, burst mode (default: 3)")
    p.add_argument("--start-step",      type=int,   default=1, choices=range(1, 9),
                   metavar="N", help="Resume from step N (1-8)")
    p.add_argument("--run-dir",         type=Path,  default=None)
    p.add_argument("--registry",        type=Path,  default=Path("./subclass_registry.json"))
    p.add_argument("--entity-registry", type=Path,  default=Path("./entity_registry.json"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.start_step > 1 and args.run_dir is None:
        print("ERROR: --run-dir is required when --start-step > 1")
        sys.exit(1)

    cfg = {
        "endpoint":            args.endpoint,
        "min_entities":        args.min_entities,
        "limit":               args.limit,
        "min_desc":            args.min_desc,
        "max_total":           args.max_total,
        "delay":               args.delay,
        "wiki_workers":        args.wiki_workers,
        "registry_path":       args.registry,
        "entity_registry_path": args.entity_registry,
    }

    run_pipeline(cfg, start_step=args.start_step, run_dir=args.run_dir)
