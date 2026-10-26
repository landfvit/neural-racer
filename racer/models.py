"""Finite JSON model files without executable serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .networks import INPUT_NAMES, get_architecture


def validate_model(model):
    if isinstance(model, dict) and model.get("format") == "neural-racer-policy" and type(model.get("version")) is int and model["version"] == 1:
        raise ValueError("Version 1 models use speed as an input. Train and save a new version 2 model with target speed as an output.")
    if not isinstance(model, dict) or model.get("format") != "neural-racer-policy" or type(model.get("version")) is not int or model["version"] != 2:
        raise ValueError("Choose a supported neural racer JSON model.")
    specification = get_architecture(model.get("architecture_id", "compact"))
    hidden = list(specification.hidden)
    expected = {"inputs": len(INPUT_NAMES), "hidden": hidden[0] if len(hidden) == 1 else hidden,
                "outputs": len(specification.outputs), "activation": "tanh"}
    if model.get("architecture") != expected:
        raise ValueError("The model architecture must match the selected supported network.")
    if model.get("inputs") != list(INPUT_NAMES):
        raise ValueError("The model inputs must match the five wall distance sensors.")
    if model.get("outputs") != list(specification.outputs):
        raise ValueError("The model outputs must match the selected network's driving controls.")
    weights = model.get("weights")
    if not isinstance(weights, list) or len(weights) != specification.parameters or any(type(value) not in (int, float) for value in weights):
        raise ValueError(f"The model must contain {specification.parameters} numeric weights.")
    try:
        array = np.asarray(weights, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Model weights must be finite numbers between -8 and 8.") from exc
    if not np.isfinite(array).all() or (np.abs(array) > 8).any():
        raise ValueError("Model weights must be finite numbers between -8 and 8.")
    return {**model, "architecture_id": specification.id, "weights": array.tolist()}


def save_model(path, model):
    Path(path).write_text(json.dumps(validate_model(model), indent=2, allow_nan=False) + "\n")


def load_model(path):
    file = Path(path)
    if file.stat().st_size > 64 * 1024:
        raise ValueError("Model files are limited to 64 KiB.")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Model JSON keys must be unique.")
            result[key] = value
        return result

    try:
        value = json.loads(file.read_text(), object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Choose a valid JSON model file.") from exc
    return validate_model(value)
