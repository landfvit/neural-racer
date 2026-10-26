"""Rebuild wider circuit-inspired practice layouts from the bundled source data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "monza": "it-1922",
    "silverstone": "gb-1948",
    "spa": "be-1925",
    "monaco": "mc-1929",
    "interlagos": "br-1940",
}
WIDTH, HEIGHT, MARGIN = 960, 640, 45
ROAD_WIDTH = 32.0
REFERENCE_SPACING, ROUTE_SPACING = 5.0, 14.0
ITERATIONS, SEPARATION_MARGIN = 350, 9.0
REPULSION, SMOOTHING, SOURCE_TETHER = 0.22, 0.25, 0.002


def project(coordinates):
    #fit the source with a uniform projection before gently adapting its bends
    radians = np.deg2rad(np.asarray(coordinates, dtype=float))
    center = radians.mean(axis=0)
    local = (radians - center) * 6_371_000
    local[:, 0] *= np.cos(center[1])
    local[:, 1] *= -1
    best = None
    for angle in np.linspace(0, np.pi, 721):
        cosine, sine = np.cos(angle), np.sin(angle)
        rotated = local @ np.array([[cosine, -sine], [sine, cosine]]).T
        extent = np.ptp(rotated, axis=0)
        scale = min((WIDTH - 2 * MARGIN) / extent[0],
                    (HEIGHT - 2 * MARGIN) / extent[1])
        if best is None or scale > best[0]:
            best = scale, rotated
    scale, rotated = best
    midpoint = (rotated.min(axis=0) + rotated.max(axis=0)) / 2
    return (rotated - midpoint) * scale + (WIDTH / 2, HEIGHT / 2), float(scale)


def resample(route, spacing):
    vectors = np.roll(route, -1, axis=0) - route
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(vectors, axis=1))]
    count = max(4, int(np.ceil(cumulative[-1] / spacing)))
    distances = np.arange(count) * cumulative[-1] / count
    return np.column_stack([
        np.interp(distances, cumulative, np.r_[route[:, axis], route[0, axis]])
        for axis in (0, 1)
    ])


def adapt(route):
    reference = resample(route, REFERENCE_SPACING)
    adapted = reference.copy()
    count = len(adapted)
    indices = np.arange(count)
    difference = np.abs(indices[:, None] - indices)
    local = np.minimum(difference, count - difference) * REFERENCE_SPACING < ROAD_WIDTH * 1.7
    for _ in range(ITERATIONS):
        delta = adapted[:, None, :] - adapted[None, :, :]
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        force = np.maximum(0, ROAD_WIDTH + SEPARATION_MARGIN - distance) / np.maximum(distance, 1e-6)
        force[local] = 0
        #separate nearby stretches while smoothing small zigzags and retaining the silhouette
        repel = np.sum(delta * force[:, :, None], axis=1)
        repel /= np.maximum((force > 0).sum(axis=1)[:, None], 1)
        smooth = (np.roll(adapted, 1, axis=0) + np.roll(adapted, -1, axis=0)) / 2 - adapted
        adapted += REPULSION * repel + SMOOTHING * smooth + SOURCE_TETHER * (reference - adapted)
    result = np.round(resample(adapted, ROUTE_SPACING), 4)
    if not 4 <= len(result) <= 256 or not np.isfinite(result).all():
        raise ValueError("The adapted circuit needs between 4 and 256 finite route points.")
    return result


def main():
    circuits = {}
    for identifier, source in SOURCES.items():
        document = json.loads((ROOT / "racer" / "data" / "source" / f"{source}.geojson").read_text())
        coordinates = np.asarray(document["features"][0]["geometry"]["coordinates"], dtype=float)
        if np.array_equal(coordinates[0], coordinates[-1]):
            coordinates = coordinates[:-1]
        route, scale = project(coordinates)
        circuits[identifier] = {"source": source, "source_scale": scale, "route": adapt(route).tolist()}
    document = {
        "version": 1,
        "road_width": ROAD_WIDTH,
        "description": "Wider circuit-inspired practice layouts with relaxed bends and separated road stretches. These approximate the source circuits.",
        "generation": {
            "reference_spacing": REFERENCE_SPACING, "route_spacing": ROUTE_SPACING,
            "iterations": ITERATIONS, "separation_margin": SEPARATION_MARGIN,
            "repulsion": REPULSION, "smoothing": SMOOTHING, "source_tether": SOURCE_TETHER,
        },
        "circuits": circuits,
    }
    destination = ROOT / "racer" / "data" / "circuit-layouts.json"
    destination.write_text(json.dumps(document, separators=(",", ":")) + "\n")
    print(f"Rebuilt {len(circuits)} practice circuits in {destination}")


if __name__ == "__main__":
    main()
