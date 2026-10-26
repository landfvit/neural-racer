"""Wide training circuits adapted from offline Formula 1 centerlines."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .tracks import Track, VehicleProfile


_LAYOUTS = Path(__file__).with_name("data") / "circuit-layouts.json"
_CATALOG = (
    ("monza", "Monza", "Fast"),
    ("silverstone", "Silverstone", "Flowing"),
    ("spa", "Spa-Francorchamps", "Technical"),
    ("monaco", "Monaco", "Extreme"),
    ("interlagos", "Interlagos", "Technical"),
)
_WIDTH, _HEIGHT = 960, 640


def circuit_catalog():
    return [{"id": key, "name": name, "difficulty": difficulty}
            for key, name, difficulty in _CATALOG]


def _road_mask(route, road_width):
    #one continuous road follows the adapted centerline with a constant width
    mask = np.zeros((_HEIGHT, _WIDTH), dtype=bool)
    radius = road_width / 2
    for start, end in zip(route, np.roll(route, -1, axis=0)):
        lower = np.maximum(np.floor(np.minimum(start, end) - radius - 1).astype(int), 0)
        upper = np.minimum(np.ceil(np.maximum(start, end) + radius + 1).astype(int), (_WIDTH, _HEIGHT))
        grid_y, grid_x = np.mgrid[lower[1]:upper[1], lower[0]:upper[0]]
        points = np.stack((grid_x + 0.5, grid_y + 0.5), axis=-1)
        delta = end - start
        fraction = np.clip(np.sum((points - start) * delta, axis=-1) / np.dot(delta, delta), 0, 1)
        nearest = start + fraction[..., None] * delta
        distance = np.sum((points - nearest) ** 2, axis=-1)
        mask[lower[1]:upper[1], lower[0]:upper[0]] |= distance <= radius * radius
    return mask


@lru_cache(maxsize=5)
def _built_circuit(identifier):
    _, name, _ = next(item for item in _CATALOG if item[0] == identifier)
    layouts = json.loads(_LAYOUTS.read_text())
    layout = layouts["circuits"][identifier]
    route = np.asarray(layout["route"], dtype=float)
    road_width = layouts["road_width"]
    mask = _road_mask(route, road_width)
    image = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    image[:] = (22, 27, 31)
    image[mask] = (224, 228, 232)
    heading = np.arctan2(*(route[1] - route[0])[::-1])
    #retain the driving physics while widening checkpoint spans to match the road
    vehicle = replace(VehicleProfile.scaled(layout["source_scale"]), gate_half_width=road_width / 2)
    return Track(mask, image, route, route[0], heading, name,
                 vehicle=vehicle)


def get_circuit(identifier):
    if not isinstance(identifier, str) or identifier not in {item[0] for item in _CATALOG}:
        raise ValueError("Choose Monza, Silverstone, Spa-Francorchamps, Monaco, or Interlagos.")
    return _built_circuit(identifier)


def export_circuit(identifier, path):
    track = get_circuit(identifier)
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Circuit images must use the .png extension.")
    buffer = BytesIO()
    Image.fromarray(track.image).save(buffer, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(buffer.getvalue())
    return path
