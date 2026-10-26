import json

import numpy as np
import pytest

from racer.engine import HIDDEN, INPUTS, LAP_LIMIT, MAX_STEPS, OUTPUTS, PARAMETERS, SIMULATION_HZ, Trainer, network, policy
from racer.models import load_model, save_model, validate_model
from racer.tracks import Track, sample_track


def open_track(mask=None):
    if mask is None:
        mask = np.ones((200, 200), dtype=bool)
    height, width = mask.shape
    route = np.array([[30., 40.], [170., 40.], [170., 170.], [30., 170.]])
    return Track(mask, np.full((height, width, 3), 255, dtype=np.uint8), route, route[0], 0., "Test")


def finish(trainer):
    while not trainer.done:
        trainer.tick(128)
    return trainer.history[-1]


def constant(target_speed=0.9, steering=0, brake=0):
    weights = np.zeros(PARAMETERS)
    weights[-OUTPUTS:] = np.arctanh([steering, target_speed, brake])
    return weights


def test_plain_network_shape_and_forward_pass():
    assert (INPUTS, HIDDEN, OUTPUTS, PARAMETERS) == (5, 12, 3, 111)
    weights = np.random.default_rng(1).normal(0, .3, (4, PARAMETERS))
    observations = np.random.default_rng(2).random((4, 5))
    result = policy(weights, observations)
    for index in range(4):
        parts = network(weights[index])
        hidden = np.tanh(observations[index] @ parts["input_hidden"] + parts["hidden_bias"])
        expected = np.tanh(hidden @ parts["hidden_output"] + parts["output_bias"])
        assert result[index] == pytest.approx(expected)


@pytest.mark.parametrize("weights,inputs", [
    (np.zeros(PARAMETERS - 1), np.zeros((1, 5))), (np.full(PARAMETERS, np.nan), np.zeros((1, 5))),
    (np.zeros(PARAMETERS), np.zeros((1, 6))), (np.zeros(PARAMETERS), np.full((1, 5), np.inf)),
])
def test_invalid_network_values_are_rejected(weights, inputs):
    with pytest.raises(ValueError):
        policy(weights, inputs)


def test_fresh_network_starts_at_zero_and_mutations_are_seeded():
    first, second = Trainer(open_track(), 16, 5), Trainer(open_track(), 16, 5)
    assert not first.weights.any() and not second.weights.any()
    assert finish(first) == finish(second)
    assert first.evolve() and second.evolve()
    np.testing.assert_array_equal(first.weights, second.weights)
    assert len(np.unique(first.weights, axis=0)) > 1
    first.tick(25)
    second.tick(25)
    assert first.snapshot() == second.snapshot()


def test_car_moves_forward_and_collision_stops_it():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant()
    trainer.tick(40)
    assert np.all(trainer.cars.x > 30)
    assert np.all(trainer.cars.speed >= 0)
    assert np.all(trainer.cars.speed <= trainer.max_speed)
    finish(trainer)
    assert trainer.cars.crashed.all()
    before = trainer.snapshot()
    trainer.tick(128)
    assert trainer.snapshot() == before
    assert len(trainer.history) == 1


def test_negative_pedals_do_not_accelerate_or_reverse():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=-.9, brake=-.9)
    trainer.tick(100)
    assert np.all(trainer.cars.speed == 0)
    assert np.all(trainer.cars.x == 30)
    assert np.all(trainer.cars.heading == 0)


def test_zero_target_automatically_brakes_to_a_stop_without_explicit_brake():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=0)
    trainer.cars.speed[:] = trainer.max_speed * .4
    before = trainer.cars.speed.copy()
    trainer.tick()
    expected = before - trainer.scale * (.00080 + .000025)
    np.testing.assert_allclose(trainer.cars.speed, expected)
    assert np.all(trainer.cars.brake == 1)
    assert np.all(trainer.cars.throttle == 0)
    trainer.tick(20)
    assert trainer.cars.alive.all()
    assert not trainer.cars.crashed.any()
    assert np.all(trainer.cars.speed == 0)
    stopped = trainer.cars.x.copy()
    trainer.tick(20)
    np.testing.assert_array_equal(trainer.cars.x, stopped)


@pytest.mark.parametrize("target_fraction", [.05, .25, .5, .9])
def test_target_speed_converges_with_bounded_acceleration_and_deceleration(target_fraction):
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=target_fraction)
    target = trainer.max_speed * target_fraction
    speeds = [trainer.cars.speed.copy()]
    for _ in range(30):
        trainer.tick()
        speeds.append(trainer.cars.speed.copy())
    changes = np.diff(speeds, axis=0)
    assert trainer.cars.alive.all()
    assert (changes >= -1e-12).all()
    assert (changes <= trainer.scale * (.00032 - .000025) + 1e-12).all()
    assert (np.asarray(speeds) <= target + 1e-12).all()
    np.testing.assert_allclose(trainer.cars.speed, target)
    trainer.weights[:] = constant(target_speed=target_fraction / 2)
    speeds = [trainer.cars.speed.copy()]
    for _ in range(15):
        trainer.tick()
        speeds.append(trainer.cars.speed.copy())
    changes = np.diff(speeds, axis=0)
    assert (changes <= 1e-12).all()
    assert (changes >= -trainer.scale * (.00080 + .000025) - 1e-12).all()
    assert (np.asarray(speeds) >= target / 2 - 1e-12).all()
    np.testing.assert_allclose(trainer.cars.speed, target / 2)
    assert trainer.snapshot()["controls"]["target_speed"] == pytest.approx(target_fraction / 2)


def test_brake_at_low_speed_stops_without_reversing_or_turning():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=0, steering=.9, brake=.99)
    trainer.cars.speed[:] = trainer.max_speed * .01
    trainer.tick(100)
    assert np.all(trainer.cars.speed == 0)
    assert np.all(trainer.cars.x == 30)
    assert np.all(trainer.cars.y == 40)
    assert np.all(trainer.cars.heading == 0)


def test_explicit_brake_overrides_an_accelerating_target_speed():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=.9, brake=.75)
    trainer.cars.speed[:] = trainer.max_speed * .5
    before = trainer.cars.speed.copy()
    trainer.tick()
    assert np.all(trainer.cars.speed < before)
    expected = before + trainer.scale * (.00032 * (1 - .75) - .00080 * .75 - .000025)
    np.testing.assert_allclose(trainer.cars.speed, expected)
    snapshot = trainer.snapshot()
    assert snapshot["controls"] == pytest.approx({"steering": 0, "target_speed": .9, "throttle": .25, "brake": .75})
    assert all(car["target_speed"] == pytest.approx(.9) and car["throttle"] == pytest.approx(.25) and car["brake"] == pytest.approx(.75)
               for car in snapshot["cars"])


def test_negative_brake_output_releases_the_pedal():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=.9, brake=-.8)
    trainer.tick()
    assert np.all(trainer.cars.speed > 0)
    assert trainer.snapshot()["controls"] == pytest.approx({"steering": 0, "target_speed": .9, "throttle": 1, "brake": 0})


def test_inactive_cars_do_not_move_under_throttle_or_braking():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=.9, steering=.8, brake=.4)
    trainer.cars.alive[:3] = False
    trainer.cars.crashed[0] = True
    trainer.cars.finished[1] = True
    trainer.cars.finish_steps[1] = 10
    trainer._update_fitness()
    before = [trainer.snapshot()["cars"][index] for index in range(3)]
    trainer.tick(20)
    assert trainer.snapshot()["cars"][:3] == before
    assert trainer.cars.elapsed_steps[3] == 20


def test_time_limit_releases_pedals_and_stops_all_cars():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=.9, brake=.1)
    trainer.cars.speed[:] = trainer.max_speed * .5
    trainer.step_count = MAX_STEPS - 1
    trainer._last_gain[:] = MAX_STEPS - 1
    trainer.tick()
    assert trainer.done and trainer.step_count == MAX_STEPS
    assert not trainer.cars.alive.any()
    assert np.all(trainer.cars.speed == 0)
    assert np.all(trainer.cars.target_speed == 0)
    assert np.all(trainer.cars.throttle == 0)
    assert np.all(trainer.cars.brake == 0)
    snapshot = trainer.snapshot()
    assert snapshot["controls"] == {"steering": 0, "target_speed": 0, "throttle": 0, "brake": 0}
    assert all(car["speed"] == car["target_speed"] == car["throttle"] == car["brake"] == 0 for car in snapshot["cars"])
    trainer.tick(20)
    assert trainer.snapshot() == snapshot


def test_one_pixel_wall_cannot_be_tunneled_at_maximum_speed():
    mask = np.ones((600, 900), dtype=bool)
    mask[:, 300] = False
    route = np.array([[100., 200.], [700., 200.], [700., 450.], [100., 450.]])
    track = Track(mask, np.full((600, 900, 3), 255, dtype=np.uint8), route, route[0], 0., "Wall")
    trainer = Trainer(track, 16)
    trainer.weights[:] = constant()
    trainer.cars.speed[:] = trainer.max_speed
    trainer.tick(100)
    assert trainer.cars.crashed.all()
    assert np.all(trainer.cars.x + trainer.car_length / 2 < 301)


def test_sensors_match_current_car_and_neural_activity():
    trainer = Trainer(open_track(), 16)
    trainer.tick(8)
    snapshot = trainer.snapshot()
    index = snapshot["best_index"]
    assert len(snapshot["sensors"]["endpoints"]) == 5
    assert len(snapshot["brain"]["inputs"]) == 5
    assert snapshot["brain"]["inputs"] == snapshot["sensors"]["values"]
    assert policy(trainer.weights[index], [snapshot["brain"]["inputs"]])[0] == pytest.approx(snapshot["brain"]["outputs"])
    json.dumps(snapshot, allow_nan=False)


def test_population_composition_does_not_change_elite_physics():
    first, second = Trainer(open_track(), 16), Trainer(open_track(), 16)
    champion = constant(target_speed=.4, steering=.1)
    first.weights[0] = second.weights[0] = champion
    first.weights[1:] = constant(target_speed=.9)
    second.weights[1:] = constant(target_speed=-.9)
    first.tick(80)
    second.tick(80)
    for field in ("x", "y", "heading", "speed", "progress", "fitness"):
        assert getattr(first.cars, field)[0] == getattr(second.cars, field)[0]


def test_checkpoint_oscillation_cannot_farm_laps_or_unbounded_progress():
    trainer = Trainer(open_track(), 16)
    gate = trainer.gates[1]
    normal = trainer.normals[1]
    before = np.repeat((gate - normal)[None], 16, axis=0)
    after = np.repeat((gate + normal)[None], 16, axis=0)
    for _ in range(200):
        trainer._advance_progress(before, after, np.ones(16, dtype=bool))
        trainer._advance_progress(after, before, np.ones(16, dtype=bool))
    assert np.all(trainer.cars.laps == 0)
    assert np.all(trainer._passed <= 1)
    assert np.all(trainer.cars.progress <= trainer.gate_spacing * 2)


def test_ordered_lap_required_and_backward_lap_does_not_count():
    forward, backward = Trainer(open_track(), 16), Trainer(open_track(), 16)
    for trainer, points in ((forward, forward.gates), (backward, backward.gates[np.r_[0, np.arange(len(backward.gates) - 1, 0, -1)]])):
        for index in range(len(points)):
            before = np.repeat(points[index][None], 16, axis=0)
            after = np.repeat(points[(index + 1) % len(points)][None], 16, axis=0)
            trainer._advance_progress(before, after, np.ones(16, dtype=bool))
    assert np.all(forward.cars.laps == 1)
    assert np.all(backward.cars.laps == 0)


def test_evolution_preserves_best_weights_and_history():
    trainer = Trainer(open_track(), 16)
    with pytest.raises(ValueError):
        trainer.evolve()
    finish(trainer)
    saved = trainer.best_weights.copy()
    history = [dict(entry) for entry in trainer.history]
    assert trainer.evolve()
    assert trainer.generation == 1
    assert np.array_equal(trainer.weights[0], saved)
    finish(trainer)
    assert trainer.history[0] == history[0]
    assert trainer.history[-1]["best_fitness"] >= history[0]["best_fitness"] - 1e-9


def test_model_round_trip_and_loaded_policy_consistency(tmp_path):
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = np.random.default_rng(23).normal(0, 0.2, trainer.weights.shape)
    model = trainer.model()
    path = tmp_path / "brain.json"
    save_model(path, model)
    restored = load_model(path)
    other = Trainer(open_track(), 16)
    other.load_model(restored)
    assert np.array_equal(other.weights[0], model["weights"])
    assert np.array_equal(other.weights[0], other.weights[-1])
    assert restored["architecture"] == {"inputs": 5, "hidden": 12, "outputs": 3, "activation": "tanh"}
    assert restored["objective"] == "minimum_lap_time"
    assert restored["simulation_hz"] == SIMULATION_HZ == 30
    assert restored["lap_limit"] == LAP_LIMIT == 1


@pytest.mark.parametrize("change", [
    {"weights": [0] * (PARAMETERS - 1)}, {"weights": [True] * PARAMETERS}, {"weights": [float("nan")] * PARAMETERS},
    {"weights": [9] * PARAMETERS}, {"architecture": {"inputs": 6}}, {"version": True},
    {"weights": [10 ** 400] * PARAMETERS},
])
def test_invalid_models_are_rejected(change):
    model = Trainer(open_track(), 16).model()
    with pytest.raises(ValueError):
        validate_model({**model, **change})


def test_sample_track_learning_completes_ordered_laps():
    trainer = Trainer(sample_track(), 64, 42)
    first = finish(trainer)
    for _ in range(15):
        trainer.evolve()
        finish(trainer)
    assert trainer.history[-1]["best_fitness"] > first["best_fitness"]
    assert trainer.history[-1]["best_laps"] >= 1
    assert np.all(np.diff([item["best_fitness"] for item in trainer.history]) >= -1e-9)
    assert trainer.step_count <= MAX_STEPS
    assert trainer.history[-1]["best_laps"] <= LAP_LIMIT


def near_finish():
    trainer = Trainer(open_track(), 16)
    trainer.weights[:] = constant(target_speed=-.9)
    trainer.weights[0] = constant()
    normal = trainer.normals[0]
    position = trainer.gates[0] - normal * trainer.max_speed * .25
    trainer.cars.x[0], trainer.cars.y[0] = position
    trainer.cars.heading[0] = np.arctan2(normal[1], normal[0])
    trainer.cars.speed[0] = trainer.max_speed
    trainer._passed[0] = len(trainer.gates) - 1
    trainer.cars.progress[0] = trainer.route_length - trainer.gate_spacing / 4
    return trainer


def test_every_finisher_beats_partial_progress_and_faster_laps_rank_first():
    trainer = Trainer(open_track(), 16)
    trainer.cars.progress[:3] = trainer.route_length
    trainer.cars.finished[1:3] = True
    trainer.cars.finish_steps[1:3] = [MAX_STEPS, MAX_STEPS - 1]
    trainer._update_fitness()
    assert trainer.cars.fitness[0] < 1000
    assert trainer.cars.fitness[1] == 1000
    assert trainer.cars.fitness[2] > trainer.cars.fitness[1] > trainer.cars.fitness[0]


def test_finish_time_is_captured_once_and_frozen_while_other_cars_continue():
    trainer = near_finish()
    trainer.tick()
    assert trainer.cars.finished[0]
    assert not trainer.cars.alive[0]
    assert trainer.cars.finish_steps[0] == 1
    before = trainer.snapshot()["cars"][0]
    assert before["lap_time"] == before["elapsed_time"] == 1 / SIMULATION_HZ
    assert trainer.snapshot()["metrics"]["best_ever_time"] == before["lap_time"]
    trainer.tick(20)
    assert trainer.snapshot()["cars"][0] == before
    assert trainer.cars.elapsed_steps[1] == 21
    assert trainer.cars.finish_steps[1] == -1


def test_viewer_step_batch_size_does_not_change_simulated_lap_timing():
    slow, fast = near_finish(), near_finish()
    while not slow.done:
        slow.tick(1)
    while not fast.done:
        fast.tick(128)
    assert slow.snapshot() == fast.snapshot()
    assert slow.history[0]["lap_time"] == 1 / SIMULATION_HZ
    assert slow.snapshot()["cars"][1]["elapsed_time"] == 150 / SIMULATION_HZ


def test_recorded_lap_time_improves_after_learning_to_complete_the_circuit():
    trainer = Trainer(sample_track(), 64, 42)
    for generation in range(31):
        finish(trainer)
        if generation < 30:
            trainer.evolve()
    laps = [entry["lap_time"] for entry in trainer.history if entry["lap_time"] is not None]
    assert len(laps) > 1
    assert laps[-1] < laps[0]
    assert np.all(np.diff(laps) <= 0)
    assert trainer.best_ever_time == min(laps)
    assert trainer.model()["lap_time"] == min(laps)
    assert trainer.history[-1]["best_laps"] == 1
    replayed = Trainer(sample_track(), 16, 42)
    replayed.load_model(trainer.model())
    assert finish(replayed)["lap_time"] == min(laps)


def test_wall_observations_and_policy_outputs_do_not_depend_on_actual_speed():
    trainer = Trainer(open_track(), 16)
    trainer.cars.speed[:] = np.linspace(0, trainer.max_speed, trainer.population)
    trainer.weights[:] = trainer.weights[0]
    sensors = trainer.sensors()
    assert sensors["values"].shape == (trainer.population, 5)
    np.testing.assert_array_equal(sensors["values"], np.repeat(sensors["values"][:1], trainer.population, axis=0))
    outputs = policy(trainer.weights, sensors["values"])
    np.testing.assert_array_equal(outputs, np.repeat(outputs[:1], trainer.population, axis=0))
    assert network(trainer.weights[0])["inputs"] == [
        "wall_left_60", "wall_left_30", "wall_forward", "wall_right_30", "wall_right_60",
    ]
    assert network(trainer.weights[0])["outputs"] == ["steering", "target_speed", "brake"]
