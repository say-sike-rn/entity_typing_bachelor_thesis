#!/usr/bin/env python3
"""
bert_cascade.py
----------------
Loads and runs the ModernBERT classifiers, and walks the cascade of models
from the root class (e.g. Person) down to whichever class has no further
trained BERT model available.
"""

from dataclasses import dataclass, field
from typing import Optional

YAGO_NS = "http://yago-knowledge.org/resource/"

# model_dir -> (tokenizer, model, device), so each model is loaded once per run.
_MODEL_CACHE: dict = {}


def label_to_iri(label: str) -> str:
    return YAGO_NS + label


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_bert_model(model_dir: str, device: Optional[str] = None):
    """Load (and cache) a tokenizer + sequence-classification model from disk."""
    if model_dir in _MODEL_CACHE:
        return _MODEL_CACHE[model_dir]

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    resolved_device = device or _pick_device()

    print(f"   [BERT] Loading model from {model_dir} onto {resolved_device} …")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(resolved_device)
    model.eval()

    entry = (tokenizer, model, resolved_device)
    _MODEL_CACHE[model_dir] = entry
    return entry


def classify_with_bert(model_dir: str, text: str, device: Optional[str] = None):
    """
    Run a single BERT classifier on `text`.

    Returns (label, confidence, all_scores) where all_scores is a
    {label: probability} dict covering every class the model knows about.
    """
    import torch

    tokenizer, model, resolved_device = load_bert_model(model_dir, device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(resolved_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)

    top_idx = int(torch.argmax(probs).item())
    label = model.config.id2label[top_idx]
    confidence = float(probs[top_idx].item())
    all_scores = {model.config.id2label[i]: float(p) for i, p in enumerate(probs)}
    return label, confidence, all_scores


@dataclass
class CascadeNode:
    """One BERT model in the cascade, plus which of its predicted labels lead
    into a further, more specific BERT model."""
    model_dir: str
    next: dict = field(default_factory=dict)  # label -> CascadeNode


def run_bert_cascade(root: CascadeNode, text: str, device: Optional[str] = None):
    """
    Walk the BERT cascade starting at `root`, descending into a child model
    whenever the predicted label has one, until reaching a label with no
    further BERT model.

    Returns (steps, final_label) where each step is:
        {"model_dir", "label", "confidence"}
    """
    steps = []
    node = root
    while True:
        label, confidence, _all_scores = classify_with_bert(node.model_dir, text, device)
        steps.append({
            "model_dir": node.model_dir,
            "label": label,
            "confidence": confidence,
        })
        child = node.next.get(label)
        if child is None:
            return steps, label
        node = child
