import copy
from dataclasses import FrozenInstanceError
import json

import numpy as np
import pytest

from racer.engine import Trainer, network, policy
from racer.models import load_model, save_model, validate_model
from racer.networks import architecture_catalog, get_architecture
from racer.tracks import Track


IDS = ("compact", "wide", "deep", "split")
SIZES = ((5, 12, 3), (5, 24, 3), (5, 16, 12, 3), (5, 20, 12, 4))
PARAMETERS = (111, 219, 339, 424)


def open_track():
    route = np.array([[100., 100.], [700., 100.], [700., 500.], [100., 500.]])
    return Track(np.ones((600, 800), dtype=bool), np.full((600, 800, 3), 255, dtype=np.uint8),
                 route, route[0], 0., "Test")


def finish(trainer):
    while not trainer.done:
        trainer.tick(128)
    return trainer.history[-1]


def constant_actions(identifier, outputs):
    specification = get_architecture(identifier)
    weights = np.zeros(specification.parameters)
    weights[-len(specification.outputs):] = np.arctanh(outputs)
    return weights


def test_architecture_catalog_has_four_distinct_immutable_choices():
    catalog = architecture_catalog()
    assert [item["id"] for item in catalog] == list(IDS)
    for identifier, sizes, parameters in zip(IDS, SIZES, PARAMETERS):
        specification = get_architecture(identifier)
        assert specification.sizes == sizes
        assert specification.parameters == parameters
        assert get_architecture(specification) is specification
        with pytest.raises(FrozenInstanceError):
            specification.id = "changed"
    catalog[0]["sizes"][1] = 999
    catalog[0]["outputs"].append("unused")
    assert architecture_catalog()[0]["sizes"] == [5, 12, 3]
    assert architecture_catalog()[0]["outputs"] == ["steering", "target_speed", "brake"]


@pytest.mark.parametrize("identifier", ["unknown", "", "../model", None, [], {}, 1, True])
def test_unknown_architectures_are_rejected(identifier):
    with pytest.raises(ValueError):
        get_architecture(identifier)
    with pytest.raises(ValueError):
        Trainer(open_track(), 16, architecture=identifier)


@pytest.mark.parametrize("identifier,sizes,parameters", tuple(zip(IDS, SIZES, PARAMETERS)))
def test_each_forward_layer_matches_independent_matrix_math(identifier, sizes, parameters):
    weights = np.random.default_rng(17).normal(0, .2, (5, parameters))
    observations = np.random.default_rng(18).uniform(0, 1, (5, 5))
    result = policy(weights, observations, identifier)
    assert result.shape == (5, sizes[-1])
    for vector, inputs, actual in zip(weights, observations, result):
        exported = network(vector, identifier)
        assert exported["id"] == identifier
        assert exported["sizes"] == list(sizes)
        assert exported["parameters"] == parameters
        assert len(exported["layers"]) == len(sizes) - 1
        offset = 0
        expected = inputs
        for previous, current, layer in zip(sizes, sizes[1:], exported["layers"]):
            matrix = vector[offset:offset + previous * current].reshape(previous, current)
            offset += previous * current
            bias = vector[offset:offset + current]
            offset += current
            np.testing.assert_array_equal(layer["weights"], matrix)
            np.testing.assert_array_equal(layer["bias"], bias)
            expected = np.tanh(expected @ matrix + bias)
        assert offset == parameters
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("identifier", IDS)
def test_every_serialized_parameter_can_change_the_driving_policy(identifier):
    parameters = get_architecture(identifier).parameters
    baseline = np.full(parameters, .025)
    inputs = np.full((1, 5), .3)
    original = policy(baseline, inputs, identifier)[0]
    changed = np.repeat(baseline[None], parameters, axis=0)
    changed[np.arange(parameters), np.arange(parameters)] += .15
    outputs = policy(changed, np.repeat(inputs, parameters, axis=0), identifier)
    assert (np.max(np.abs(outputs - original), axis=1) > 1e-7).all()


@pytest.mark.parametrize("identifier", IDS)
@pytest.mark.parametrize("problem", ["short", "long", "nan", "infinity", "bound", "empty", "bad_inputs", "infinite_inputs"])
def test_architecture_specific_shapes_and_finite_values_are_enforced(identifier, problem):
    parameters = get_architecture(identifier).parameters
    weights, inputs = np.zeros(parameters), np.zeros((1, 5))
    if problem == "short":
        weights = weights[:-1]
    elif problem == "long":
        weights = np.zeros(parameters + 1)
    elif problem == "nan":
        weights[-1] = np.nan
    elif problem == "infinity":
        weights[0] = np.inf
    elif problem == "bound":
        weights[-1] = 8.01
    elif problem == "empty":
        weights, inputs = np.zeros((0, parameters)), np.zeros((0, 5))
    elif problem == "bad_inputs":
        inputs = np.zeros((1, 6))
    elif problem == "infinite_inputs":
        inputs[0, 0] = np.inf
    with pytest.raises(ValueError):
        policy(weights, inputs, identifier)


@pytest.mark.parametrize("identifier", IDS)
def test_activity_contains_the_real_activation_of_every_hidden_layer(identifier):
    trainer = Trainer(open_track(), 16, 31, architecture=identifier)
    trainer.weights[:] = np.random.default_rng(31).normal(0, 0.2, trainer.weights.shape)
    trainer.tick(6)
    state = trainer.snapshot()
    specification = trainer.architecture
    assert state["network"]["id"] == identifier
    assert [len(values) for values in state["brain"]["hidden_layers"]] == list(specification.hidden)
    assert state["brain"]["hidden"] == state["brain"]["hidden_layers"][0]
    current = np.asarray(state["brain"]["inputs"])
    expected_layers = []
    for layer in state["network"]["layers"]:
        current = np.tanh(current @ np.asarray(layer["weights"]) + layer["bias"])
        expected_layers.append(current)
    np.testing.assert_allclose(state["brain"]["outputs"], expected_layers[-1])
    for actual, expected in zip(state["brain"]["hidden_layers"], expected_layers[:-1]):
        np.testing.assert_allclose(actual, expected)
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize("identifier", IDS)
def test_initialization_and_evolution_are_seeded_for_every_architecture(identifier):
    first = Trainer(open_track(), 16, 7, architecture=identifier)
    second = Trainer(open_track(), 16, 7, architecture=identifier)
    assert first.weights.shape == (16, get_architecture(identifier).parameters)
    assert not first.weights.any()
    np.testing.assert_array_equal(first.weights, second.weights)
    assert finish(first) == finish(second)
    champion = first.best_weights.copy()
    assert first.evolve() and second.evolve()
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.weights[0], champion)
    assert len(np.unique(first.weights, axis=0)) > 1
    first.tick(12)
    second.tick(12)
    assert first.snapshot() == second.snapshot()


@pytest.mark.parametrize("identifier", IDS)
def test_zero_baseline_cannot_drive_and_first_mutations_can_move(identifier):
    trainer = Trainer(open_track(), 64, 42, architecture=identifier)
    before = np.column_stack((trainer.cars.x, trainer.cars.y, trainer.cars.heading))
    brain = trainer.activity()
    assert all(not np.asarray(layer).any() for layer in brain["hidden_layers"])
    assert not np.asarray(brain["outputs"]).any()
    baseline = finish(trainer)
    np.testing.assert_array_equal(np.column_stack((trainer.cars.x, trainer.cars.y, trainer.cars.heading)), before)
    assert baseline["best_progress"] == 0 and baseline["lap_time"] is None
    assert not trainer.cars.finished.any() and not trainer.cars.crashed.any()
    assert trainer.evolve()
    trainer.tick(128)
    assert trainer.cars.progress.max() > 0
    assert np.ptp(trainer.cars.x) > 0 or np.ptp(trainer.cars.y) > 0
    assert np.isfinite(trainer.weights).all()


@pytest.mark.parametrize("operation", ["reset", "architecture", "circuit"])
def test_fresh_workbench_runs_clear_previous_connections_and_output_biases(operation):
    from racer.service import Workbench

    bench = Workbench()
    bench.trainer.weights[:] = 0.25
    if operation == "reset":
        state = bench.reset()
    elif operation == "architecture":
        state = bench.select_architecture("deep")
    else:
        state = bench.select_circuit("monaco")
    assert not bench.trainer.weights.any()
    assert state["generation"] == state["step_count"] == 0
    assert not np.asarray(state["brain"]["outputs"]).any()
    assert state["history"] == [] and state["metrics"]["best_ever_time"] is None


@pytest.mark.parametrize("identifier", IDS)
def test_model_round_trip_selects_the_correct_architecture_and_replays_its_policy(identifier, tmp_path):
    trainer = Trainer(open_track(), 16, 23, architecture=identifier)
    trainer.weights[:] = np.random.default_rng(23).normal(0, 0.2, trainer.weights.shape)
    trainer.tick(8)
    model = trainer.model()
    path = tmp_path / f"{identifier}.json"
    save_model(path, model)
    loaded = load_model(path)
    assert np.asarray(loaded["weights"]).any()
    other = Trainer(open_track(), 16, 99)
    other.load_model(loaded)
    assert other.architecture.id == identifier
    assert loaded["architecture_id"] == identifier
    np.testing.assert_array_equal(other.weights[0], model["weights"])
    np.testing.assert_array_equal(other.weights[0], other.weights[-1])
    expected = Trainer(open_track(), 16, 99, architecture=identifier)
    expected.weights[:] = model["weights"]
    expected.tick(25)
    other.tick(25)
    assert other.snapshot() == expected.snapshot()


def test_current_compact_model_without_architecture_id_still_loads():
    model = Trainer(open_track(), 16).model()
    model.pop("architecture_id")
    restored = validate_model(model)
    assert restored["architecture_id"] == "compact"
    trainer = Trainer(open_track(), 16, architecture="split")
    trainer.load_model(model)
    assert trainer.architecture.id == "compact"
    assert trainer.weights.shape == (16, 111)


@pytest.mark.parametrize("identifier", IDS)
@pytest.mark.parametrize("problem", ["wrong_id", "wrong_hidden", "wrong_outputs", "short", "boolean", "nan"])
def test_model_metadata_and_weights_cannot_disagree(identifier, problem):
    model = copy.deepcopy(Trainer(open_track(), 16, architecture=identifier).model())
    if problem == "wrong_id":
        model["architecture_id"] = "split" if identifier != "split" else "compact"
    elif problem == "wrong_hidden":
        model["architecture"]["hidden"] = [999]
    elif problem == "wrong_outputs":
        model["outputs"] = ["steering", "throttle", "decoration"]
    elif problem == "short":
        model["weights"].pop()
    elif problem == "boolean":
        model["weights"][-1] = True
    elif problem == "nan":
        model["weights"][-1] = float("nan")
    with pytest.raises(ValueError):
        validate_model(model)


def test_all_four_split_outputs_act_on_real_car_physics():
    def run(outputs):
        trainer = Trainer(open_track(), 16, architecture="split")
        trainer.weights[:] = constant_actions("split", outputs)
        trainer.cars.speed[:] = trainer.max_speed * .5
        trainer.tick()
        assert trainer.cars.alive.all()
        return trainer

    straight = run([0, 0, .9, 0])
    left = run([.7, .2, .9, 0])
    right = run([.2, .7, .9, 0])
    released = run([-.8, -.9, .9, 0])
    stopped_target = run([0, 0, 0, 0])
    braked = run([0, 0, .9, .75])
    assert np.all(left.cars.heading < straight.cars.heading)
    assert np.all(right.cars.heading > straight.cars.heading)
    np.testing.assert_allclose(left.cars.heading, -right.cars.heading)
    np.testing.assert_allclose(released.cars.heading, straight.cars.heading)
    assert np.all(straight.cars.speed > braked.cars.speed)
    assert np.all(braked.cars.speed > stopped_target.cars.speed)
    assert left.snapshot()["controls"]["steering"] == pytest.approx(-.5)
    assert right.snapshot()["controls"]["steering"] == pytest.approx(.5)
    assert braked.snapshot()["controls"] == pytest.approx({"steering": 0, "target_speed": .9, "throttle": .25, "brake": .75})
    for trainer in (straight, left, right, released, stopped_target, braked):
        assert all(car["steering"] == pytest.approx(trainer.snapshot()["controls"]["steering"])
                   for car in trainer.snapshot()["cars"])


@pytest.mark.parametrize("identifier", IDS)
def test_old_speed_input_model_is_rejected_with_migration_explanation(identifier):
    model = Trainer(open_track(), 16, architecture=identifier).model()
    assert model["version"] == 2
    model["version"] = 1
    model["architecture"]["inputs"] = 6
    model["inputs"].append("speed")
    with pytest.raises(ValueError, match="Version 1.*speed as an input.*version 2"):
        validate_model(model)
