#!/usr/bin/env python3
"""
entity_typer.py
--------------- 
LLM-based entity typing trainer for hierarchical ontologies (e.g. YAGO).
"""

import argparse
import os
import re
import textwrap
from typing import Optional


SUBCLASS_QUERY = """
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX yago:   <http://yago-knowledge.org/resource/>

SELECT DISTINCT ?child ?label WHERE {{
    ?child rdfs:subClassOf <{parent_iri}> . 
}}
ORDER BY ?child
LIMIT {limit}
"""

ENTITY_SAMPLE_QUERY = """
PREFIX rdf:    <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX schema: <http://schema.org/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>
PREFIX yago:   <http://yago-knowledge.org/resource/>

SELECT DISTINCT ?entity ?label WHERE {{
    ?entity rdf:type <{class_iri}>
}}
LIMIT {limit}
"""

PATH_CHECK_QUERY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

ASK {{
    <{entity_iri}> rdf:type ?type .
    ?type rdfs:subClassOf* <{class_iri}> .
}}
"""

def is_entity_under_class(endpoint: str, entity_iri: str, class_iri: str) -> bool:
    """Returns True if entity_iri is of a type that is a subclass* of class_iri."""
    import requests
    q = PATH_CHECK_QUERY.format(entity_iri=entity_iri, class_iri=class_iri)
    r = requests.post(
        endpoint,
        data=q.encode("utf-8"),
        headers={"Content-Type": "application/sparql-query", "Accept": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["boolean"]


def clean_iri(iri: str) -> str:
    return re.sub(r"_Q\d+$", "", iri)


def sparql_query(endpoint: str, query: str) -> list[dict]:
    import requests
    r = requests.get(
        endpoint,
        params={"query": query},
        headers={"Accept": "application/json"},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["results"]["bindings"]


def fetch_subclasses(endpoint, parent_iri, limit=1000):
    #parent_iri = clean_iri(parent_iri)
    q = SUBCLASS_QUERY.format(parent_iri=parent_iri, limit=limit)
    bindings = sparql_query(endpoint, q)
    return [
        {"iri": b["child"]["value"], "label": b.get("label", {}).get("value", _iri_to_label(b["child"]["value"]))}
        for b in bindings
    ]

# I removed entity examples a while back. 
def fetch_entity_sample(endpoint: str, class_iri: str, limit: int = 10) -> list[str]:
    #q = ENTITY_SAMPLE_QUERY.format(class_iri=class_iri, limit=limit)
    #bindings = sparql_query(endpoint, q)
    labels = []
    #for b in bindings:
    #    label = b.get("label", {}).get("value")
    #    if not label:
    #        label = _iri_to_label(b["entity"]["value"])
    #    labels.append(label)
    return labels


def _iri_to_label(iri: str) -> str:
    local = iri.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    local = re.sub(r"_u([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), local)
    return local.replace("_", " ")


def clean_label(label: str) -> str:
    """Remove Q-numbers like Q15980804 from labels."""
    return re.sub(r"\s*Q\d+$", "", label).strip()


def build_prompt(
    current_class: str,
    subclasses: list[dict],
    entity_to_classify: Optional[str] = None,
    description: Optional[str] = None,
) -> str:

    options_text = "\n".join(
        f"  {i+1}. {clean_label(sc['label'])}"
        for i, sc in enumerate(subclasses)
    )

    if entity_to_classify:
        task = textwrap.dedent(f"""
            Your task: Classify the entity "{entity_to_classify}" into exactly ONE
            of the subclasses listed below.
        """).strip()
        question = (
            f'Which subclass best describes "{entity_to_classify}"? '
            "Reply with the number and subclass name only, e.g. '5. SportsPerson'."
        )
    else:
        task = textwrap.dedent("""
            Your task: Given the current ontology class and its candidate subclasses,
            identify which subclass is the MOST SPECIFIC and semantically correct
            next step for the typical entities in this class. If you think that none
            of the options fit the description, answer with FINISHED instead of choosing a class.
        """).strip()
        question = (
            "Which single subclass is the most representative next step? "
            "Reply with the number and subclass name only, e.g. '5. SportsPerson'."
        )

    description_text = description if description else "(no description provided)"

    prompt = textwrap.dedent(f"""
        You are an expert knowledge-graph ontologist working with the YAGO dataset.

        Current ontology class : {current_class}
        Please classify this entity in one of the subclasses below, based on a short
        description of the entity and your understanding of the class hierarchy.

        Description of the entity: {description_text}

        {task}

        Candidate subclasses:
{options_text}

        {question}
    """).strip()

    return prompt


def call_gemini(prompt: str, model: str = "gemini-2.5-flash", **kwargs) -> str:
    from google import genai
    import os

    api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY env var or pass --api-key")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=kwargs.get("temperature", 0.0),
            max_output_tokens=kwargs.get("max_tokens", 128),
        ),
    )
    return response.text.strip()


def call_openai_compatible( # I never used this to be honest
    prompt: str,
    model: str = "gpt-4o-mini",
    api_base: str = "https://api.openai.com/v1",
    **kwargs,
) -> str:
    from openai import OpenAI

    api_key = kwargs.get("api_key") or os.environ.get("OPENAI_API_KEY", "sk-placeholder")
    client = OpenAI(api_key=api_key, base_url=api_base)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 0.0),
        max_tokens=kwargs.get("max_tokens", 128),
    )
    return response.choices[0].message.content.strip()


# Module-level cache so the model loads only once per session
_HF_CACHE: dict = {}

# I never used this either:
def call_huggingface(prompt: str, model: str = "Qwen/Qwen2.5-1.5B-Instruct", **kwargs) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import threading

    global _HF_CACHE

    if _HF_CACHE.get("model_name") != model:
        print(f"   [HF] Loading model {model} …")
        tokenizer = AutoTokenizer.from_pretrained(model)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model, device_map="auto", torch_dtype="auto"
        )
        _HF_CACHE = {"model_name": model, "tokenizer": tokenizer, "model": hf_model}
        print(f"   [HF] Model loaded.\n")

    tokenizer = _HF_CACHE["tokenizer"]
    hf_model  = _HF_CACHE["model"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert knowledge-graph ontologist working with YAGO. "
                "When asked to classify an entity, reply with the number and subclass name only, "
                "e.g. '5. SportsPerson'. If no subclass fits, reply with FINISHED."
            )
        },
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(hf_model.device)

    input_token_count = inputs.input_ids.shape[1]
    print(f"   [HF] Input tokens: {input_token_count}")

    result = {"output": None, "error": None}

    def generate():
        try:
            with torch.no_grad():
                output_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = output_ids[0][inputs.input_ids.shape[1]:]
            result["output"] = tokenizer.decode(generated, skip_special_tokens=True).strip()
        except Exception as e:
            result["error"] = str(e)

    thread = threading.Thread(target=generate)
    thread.start()
    thread.join(timeout=120)

    if thread.is_alive():
        print("   ⚠️  [HF] Generation timed out after 120s — skipping this entity.")
        return "something went wrong rahhhh"
    if result["error"]:
        print(f"   ⚠️  [HF] Generation error: {result['error']}")
        return "something went wrong rahhhh"

    print(f"   [HF] Output: {result['output']}")
    return result["output"]


def call_mistral(prompt: str, **kwargs) -> str:
    import requests

    model = kwargs.get("model", "mistral-small-latest")
    api_key = kwargs.get("api_key") or os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Set MISTRAL_API_KEY env var or pass --api-key")

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.0),
            "max_tokens": kwargs.get("max_tokens", 128),
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


BACKENDS = {
    "gemini": call_gemini,
    "openai": call_openai_compatible,
    "hf": call_huggingface,
    "mistral": call_mistral
}


def call_llm(backend: str, prompt: str, **kwargs) -> str:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(BACKENDS)}")
    return BACKENDS[backend](prompt, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM-based entity typing on YAGO (or similar) SPARQL endpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--iri", default="http://schema.org/Person",
                   help="IRI of the current ontology class to classify from.")
    p.add_argument("--entity", default=None,
                   help="Specific entity label/IRI to classify (optional).")
    p.add_argument("--description", default=None,
                   help="Short description of the entity to classify.")
    p.add_argument("--endpoint", default="http://localhost:9004",
                   help="SPARQL endpoint URL.")
    p.add_argument("--backend", choices=list(BACKENDS), default="gemini",
                   help="LLM backend to use.")
    p.add_argument("--model", default=None,
                   help="Model name (backend-specific).")
    p.add_argument("--api-base", default="https://api.openai.com/v1",
                   help="Base URL for OpenAI-compatible endpoints.")
    p.add_argument("--api-key", default=None,
                   help="API key (falls back to env vars).")
    p.add_argument("--subclass-limit", type=int, default=1000,
                   help="Max subclasses to fetch via SPARQL.")
    p.add_argument("--entity-sample", type=int, default=10,
                   help="Number of example entities to include in the prompt.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature (0 = deterministic).")
    p.add_argument("--max-tokens", type=int, default=128,
                   help="Max tokens for the LLM response.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the prompt without calling the LLM.")
    return p.parse_args()


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "hf": "Qwen/Qwen2.5-1.5B-Instruct",
    "mistral": "mistral-small-latest"
}


def loop(args, model, current_label):
    print("⏳ Fetching subclasses from SPARQL endpoint …")
    try:
        subclasses = fetch_subclasses(args.endpoint, args.iri, limit=args.subclass_limit)
    except Exception as e:
        print(f"⚠️  SPARQL fetch failed: {e}")
        subclasses = []

    print(f"   → {len(subclasses)} subclasses found.\n")

    print("Entity exampeles have been removed from this thesis. ")

    if not subclasses:
        print("No subclasses available — cannot build a meaningful prompt.")
        return "FINISHED"

    prompt = build_prompt(
        current_class=current_label,
        subclasses=subclasses,
        entity_to_classify=args.entity,
        description=args.description,
    )

    print("─── PROMPT ──────────────────────────────────────────────────")
    print(prompt)
    print("─────────────────────────────────────────────────────────────\n")

    if args.dry_run:
        print("[dry-run] Skipping LLM call.")
        return "something went wrong rahhhh"

    print("⏳ Calling LLM …")
    response = call_llm(
        backend=args.backend,
        prompt=prompt,
        model=model,
        api_base=args.api_base,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print("\n─── LLM RESPONSE ────────────────────────────────────────────")
    print(response)
    print("─────────────────────────────────────────────────────────────\n")

    response_lower = response.strip().lower()

    if response_lower == "finished":
        print("LLM indicated finished.")
        return "FINISHED"

    # Build a lookup of clean label → original subclass for validation
    valid = {clean_label(sc["label"]).lower(): sc for sc in subclasses}

    # Try number first (e.g. "5. SportsPerson")
    m = re.search(r"\b(\d+)\b", response)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(subclasses):
            chosen = subclasses[idx]
            print(f"✅ Chosen subclass : {clean_label(chosen['label'])}")
            print(f"   IRI            : {chosen['iri']}")
            return chosen['iri']
        else:
            print(f"⚠️  Index {idx+1} out of range ({len(subclasses)} subclasses).")

    # Fall back to label matching — only against valid subclasses
    for clean, sc in valid.items():
        if clean in response_lower:
            print(f"✅ Matched subclass : {clean_label(sc['label'])}")
            print(f"   IRI             : {sc['iri']}")
            return sc['iri']

    print(f"⚠️  Response '{response.strip()}' did not match any valid subclass.")
    return "something went wrong rahhhh"


def main():
    args = parse_args()
    model = args.model or DEFAULT_MODELS[args.backend]

    current_label = _iri_to_label(args.iri)
    print(f"\n{'='*60}")
    print(f"  Current class : {current_label}")
    print(f"  IRI           : {args.iri}")
    print(f"  Backend       : {args.backend}  |  Model: {model}")
    print(f"{'='*60}\n")

    for i in range(10):
        result = loop(args, model, current_label)
        print(result)
        if result == "something went wrong rahhhh":
            print("No valid subclass was found.")
            break
        if result == "FINISHED":
            break
        args.iri = result
        current_label = _iri_to_label(args.iri)


if __name__ == "__main__":
    main()