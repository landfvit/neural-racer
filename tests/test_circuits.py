import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import pytest

from racer.circuits import circuit_catalog, export_circuit, get_circuit
from racer.engine import MAX_STEPS, SIMULATION_HZ, Trainer
from racer.tracks import load_image
from scripts.build_circuit_layouts import project


IDS = ("monza", "silverstone", "spa", "monaco", "interlagos")
NAMES = ("Monza", "Silverstone", "Spa-Francorchamps", "Monaco", "Interlagos")
SOURCES = ("it-1922", "gb-1948", "be-1925", "mc-1929", "br-1940")
ROOT = Path(__file__).resolve().parent.parent


def _segments_cross(first, second, third, fourth):
    direction, other = second - first, fourth - third
    determinant = direction[0] * other[1] - direction[1] * other[0]
    offset = third - first
    if abs(determinant) < 1e-9:
        if abs(offset[0] * direction[1] - offset[1] * direction[0]) > 1e-9:
            return False
        axis = int(np.argmax(np.abs(direction)))
        return max(min(first[axis], second[axis]), min(third[axis], fourth[axis])) <= min(
            max(first[axis], second[axis]), max(third[axis], fourth[axis])) + 1e-9
    along = (offset[0] * other[1] - offset[1] * other[0]) / determinant
    across = (offset[0] * direction[1] - offset[1] * direction[0]) / determinant
    return -1e-9 <= along <= 1 + 1e-9 and -1e-9 <= across <= 1 + 1e-9


def test_catalog_contains_only_five_real_circuits_and_is_detached():
    catalog = circuit_catalog()
    assert [item["id"] for item in catalog] == list(IDS)
    assert [item["name"] for item in catalog] == list(NAMES)
    assert all(isinstance(item["difficulty"], str) and item["difficulty"] for item in catalog)
    catalog[0]["name"] = "Changed"
    assert circuit_catalog()[0]["name"] == "Monza"
    hashes = [hashlib.sha256(get_circuit(identifier).mask.tobytes()).hexdigest() for identifier in IDS]
    assert len(set(hashes)) == len(IDS)


def test_source_geometry_is_bundled_offline_with_pinned_provenance_and_license():
    source_root = ROOT / "racer" / "data" / "source"
    provenance = json.loads((source_root / "provenance.json").read_text())
    assert provenance["repository"] == "https://github.com/bacinger/f1-circuits"
    assert re.fullmatch(r"[0-9a-f]{40}", provenance["revision"])
    assert provenance["license"] == "MIT"
    license_text = (source_root / "LICENSE").read_text()
    assert "Tomislav Bacinger" in license_text and "Permission is hereby granted" in license_text
    for filename, digest in provenance["sha256"].items():
        assert hashlib.sha256((source_root / filename).read_bytes()).hexdigest() == digest
    for source in SOURCES:
        assert f"circuits/{source}.geojson" in provenance["files"]
        document = json.loads((source_root / f"{source}.geojson").read_text())
        assert document["type"] == "FeatureCollection" and len(document["features"]) == 1
        feature = document["features"][0]
        assert feature["properties"]["id"] == source
        assert feature["geometry"]["type"] == "LineString"
        coordinates = np.asarray(feature["geometry"]["coordinates"], dtype=float)
        assert coordinates.shape[1] == 2 and len(coordinates) >= 100
        assert np.isfinite(coordinates).all()
        np.testing.assert_array_equal(coordinates[0], coordinates[-1])


@pytest.mark.parametrize("identifier,source", tuple(zip(IDS, SOURCES)))
def test_training_routes_stay_recognizable_after_widening(identifier, source):
    document = json.loads((ROOT / "racer" / "data" / "source" / f"{source}.geojson").read_text())
    coordinates = np.asarray(document["features"][0]["geometry"]["coordinates"], dtype=float)[:-1]
    original, _ = project(coordinates)
    route = get_circuit(identifier).route

    def distance_to_route(points, centerline):
        vectors = np.roll(centerline, -1, axis=0) - centerline
        relative = points[:, None] - centerline[None]
        fractions = np.clip(np.sum(relative * vectors, axis=-1) / np.sum(vectors**2, axis=-1), 0, 1)
        return np.linalg.norm(relative - fractions[..., None] * vectors, axis=-1).min(axis=1)

    #corners may open up, while the overall layout stays close to the source in both directions
    assert distance_to_route(route, original).max() < 30
    assert distance_to_route(original, route).max() < 30
    original_length = np.linalg.norm(np.roll(original, -1, axis=0) - original, axis=1).sum()
    adapted_length = np.linalg.norm(np.roll(route, -1, axis=0) - route, axis=1).sum()
    assert 0.8 < adapted_length / original_length < 1.1


@pytest.mark.parametrize("identifier", IDS)
def test_training_roads_allow_clearance_across_the_widened_surface(identifier):
    track = get_circuit(identifier)
    vectors = np.roll(track.route, -1, axis=0) - track.route
    centers = (track.route + np.roll(track.route, -1, axis=0)) / 2
    normals = np.column_stack((-vectors[:, 1], vectors[:, 0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    #the road has room for visible cars on both sides, including the tight approaches
    points = centers[:, None] + normals[:, None] * np.linspace(-14, 14, 29)[None, :, None]
    assert track.mask[points[..., 1].astype(int), points[..., 0].astype(int)].all()
    assert track.vehicle.gate_half_width == 16
    assert track.vehicle.car_width < track.vehicle.gate_half_width / 2


@pytest.mark.parametrize("identifier", IDS)
def test_routes_are_finite_bounded_simple_closed_loops_with_safe_starts(identifier):
    track = get_circuit(identifier)
    route = track.route
    assert 4 <= len(route) <= 256
    assert track.mask.dtype == bool and track.image.dtype == np.uint8
    assert np.isfinite(route).all() and np.isfinite(track.heading)
    assert not np.array_equal(route[0], route[-1])
    np.testing.assert_array_equal(track.start, route[0])
    assert (route[:, 0] >= 0).all() and (route[:, 0] < track.width).all()
    assert (route[:, 1] >= 0).all() and (route[:, 1] < track.height).all()
    lengths = np.linalg.norm(np.roll(route, -1, axis=0) - route, axis=1)
    assert (lengths > 1e-6).all() and lengths.sum() > 80
    for index, first in enumerate(route):
        second = route[(index + 1) % len(route)]
        positions = np.linspace(first, second, int(np.ceil(lengths[index] * 2)) + 1)
        assert track.mask[positions[:, 1].astype(int), positions[:, 0].astype(int)].all()
        for following in range(index + 2, len(route)):
            if index == 0 and following == len(route) - 1:
                continue
            assert not _segments_cross(first, second, route[following], route[(following + 1) % len(route)])
    trainer = Trainer(track, 16, 42)
    assert trainer._on_road(trainer.cars.x, trainer.cars.y, trainer.cars.heading).all()
    tangents = np.roll(route, -1, axis=0) - np.roll(route, 1, axis=0)
    headings = np.arctan2(tangents[:, 1], tangents[:, 0])
    assert trainer._on_road(route[:, 0], route[:, 1], headings).all()
    first_direction = np.arctan2(*(route[1] - route[0])[::-1])
    difference = np.arctan2(np.sin(track.heading - first_direction), np.cos(track.heading - first_direction))
    assert abs(difference) < 0.1


@pytest.mark.parametrize("identifier", IDS)
def test_road_mask_forms_one_loop_without_merged_shortcuts(identifier):
    track = get_circuit(identifier)
    road = Image.fromarray(track.mask.astype(np.uint8) * 255).copy()
    ImageDraw.floodfill(road, tuple(track.start.astype(int)), 128)
    assert not (np.asarray(road) == 255).any()
    background = Image.fromarray((~track.mask).astype(np.uint8) * 255).copy()
    ImageDraw.floodfill(background, (0, 0), 128)
    enclosed = np.asarray(background) == 255
    assert enclosed.any()
    y, x = np.argwhere(enclosed)[0]
    ImageDraw.floodfill(background, (int(x), int(y)), 64)
    assert not (np.asarray(background) == 255).any()


@pytest.mark.parametrize("identifier", IDS)
def test_png_exports_are_exact_exclusive_and_checked_in(identifier, tmp_path):
    path = export_circuit(identifier, tmp_path / f"{identifier}.png")
    track = get_circuit(identifier)
    np.testing.assert_array_equal(load_image(path), track.image)
    np.testing.assert_array_equal(load_image(ROOT / "examples" / f"{identifier}.png"), track.image)
    with Image.open(path) as image:
        assert image.format == "PNG"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_circuit(identifier, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("identifier", IDS)
def test_cached_circuit_arrays_are_immutable(identifier):
    first, second = get_circuit(identifier), get_circuit(identifier)
    assert first is second
    for array in (first.mask, first.image, first.route, first.start):
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("identifier", ["../sample", "/tmp/track.png", "unknown", "", None, [], "oval", "gauntlet"])
def test_catalog_lookup_never_treats_ids_as_file_paths_or_accepts_retired_tracks(identifier):
    with pytest.raises(ValueError):
        get_circuit(identifier)


@pytest.mark.parametrize("identifier", IDS)
def test_actual_car_physics_can_complete_every_circuit_before_the_episode_limit(identifier, monkeypatch):
    import racer.engine as engine

    trainer = Trainer(get_circuit(identifier), 16, 42)
    route = trainer.track.route
    vectors = np.roll(route, -1, axis=0) - route
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.r_[0, np.cumsum(lengths)]
    perimeter = cumulative[-1]
    incoming = route - np.roll(route, 1, axis=0)
    cross = np.abs(incoming[:, 0] * vectors[:, 1] - incoming[:, 1] * vectors[:, 0])
    curvature = 2 * cross / np.maximum(
        np.linalg.norm(incoming, axis=1) * lengths * np.linalg.norm(incoming + vectors, axis=1), 1e-12)
    safe_speeds = np.minimum(trainer.max_speed * 0.65,
                            np.sqrt(trainer.max_lateral_acceleration / np.maximum(curvature, 1e-9)) * 0.65)

    def diagnostic_driver(_weights, _inputs, _architecture="compact"):
        #test-only route following checks geometry through the production physics and collisions
        cars = trainer.cars
        positions = np.column_stack((cars.x, cars.y))
        relative = positions[:, None, :] - route[None]
        fractions = np.clip(np.einsum("nsi,si->ns", relative, vectors) / lengths**2, 0, 1)
        projected = route[None] + fractions[..., None] * vectors[None]
        nearest = np.argmin(np.linalg.norm(projected - positions[:, None], axis=-1), axis=1)
        distance = cumulative[nearest] + fractions[np.arange(trainer.population), nearest] * lengths[nearest]
        ahead = np.maximum(trainer.car_length * 1.2, cars.speed * 2)
        target_distance = (distance + ahead) % perimeter
        target = np.column_stack([
            np.interp(target_distance, cumulative, np.r_[route[:, axis], route[0, axis]]) for axis in (0, 1)])
        delta = target - positions
        angle = np.arctan2(delta[:, 1], delta[:, 0]) - cars.heading
        curve = 2 * np.sin(angle) / np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        steering = np.clip(np.arctan(curve * trainer.wheelbase) / trainer.max_steer_angle, -1, 1)
        #anticipate braking before bends while retaining one controller for all five circuits
        upcoming = (cumulative[:-1][None] - distance[:, None]) % perimeter
        speed_limits = np.sqrt(safe_speeds[None]**2 + trainer.braking * upcoming)
        desired_speed = np.minimum(trainer.max_speed * 0.65, speed_limits.min(axis=1))
        return np.column_stack((steering, desired_speed / trainer.max_speed, np.zeros(trainer.population)))

    monkeypatch.setattr(engine, "policy", diagnostic_driver)
    while not trainer.done:
        trainer.tick(128)
    assert trainer.cars.finished.all()
    assert not trainer.cars.crashed.any()
    assert trainer.cars.laps.min() == trainer.cars.laps.max() == 1
    assert trainer.history[-1]["best_laps"] == 1
    assert 0 < trainer.history[-1]["lap_time"] <= MAX_STEPS / SIMULATION_HZ
