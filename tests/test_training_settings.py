import copy

import numpy as np
import pytest

import racer.engine as engine_module
from racer.circuits import get_circuit
from racer.engine import Trainer
from racer.service import Workbench


@pytest.mark.parametrize('limit', [1, 10000])
def test_generation_budget_accepts_both_boundaries(limit):
    trainer = Trainer(get_circuit('monza'), population=16, generation_limit=limit)
    assert trainer.generation_limit == limit and trainer.history == []


@pytest.mark.parametrize('limit', [0, -1, 10001, True, 1.5, '2', None])
def test_invalid_generation_budget_rejected_before_changing_trainer(limit):
    with pytest.raises(ValueError):
        Trainer(get_circuit('monza'), population=16, generation_limit=limit)
    trainer = Trainer(get_circuit('monza'), population=16)
    before = copy.deepcopy(trainer.snapshot())
    with pytest.raises(ValueError):
        trainer.set_generation_limit(limit)
    assert trainer.generation_limit == 100 and trainer.snapshot() == before


def finish_budget(trainer):
    while True:
        while not trainer.done:
            trainer.tick(128)
        if not trainer.evolve():
            break


@pytest.mark.parametrize('limit', [1, 3])
def test_single_trainer_stops_after_exactly_the_requested_completed_generations(monkeypatch, limit):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    trainer = Trainer(get_circuit('monza'), population=16, generation_limit=limit)
    finish_budget(trainer)
    assert trainer.done and trainer.generation == limit - 1
    assert [item['generation'] for item in trainer.history] == list(range(limit))
    before = copy.deepcopy(trainer.snapshot())
    trainer.tick(128)
    assert not trainer.evolve() and trainer.snapshot() == before


def test_extending_a_finished_trainer_keeps_weights_and_history(monkeypatch):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    trainer = Trainer(get_circuit('monza'), population=16, generation_limit=2)
    finish_budget(trainer)
    before, weights = copy.deepcopy(trainer.snapshot()), trainer.weights.copy()
    trainer.set_generation_limit(4)
    assert trainer.history == before['history'] and trainer.done and trainer.generation == 1
    np.testing.assert_array_equal(trainer.weights, weights)
    finish_budget(trainer)
    assert len(trainer.history) == 4 and trainer.generation == 3
    assert trainer.history[:2] == before['history']


def test_budget_reduction_keeps_the_current_attempt_and_completed_history(monkeypatch):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    trainer = Trainer(get_circuit('monza'), population=16, generation_limit=4)
    trainer.tick(2)
    assert trainer.evolve()
    before = copy.deepcopy(trainer.snapshot())
    with pytest.raises(ValueError):
        trainer.set_generation_limit(1)
    assert trainer.generation_limit == 4 and trainer.snapshot() == before
    trainer.set_generation_limit(2)
    trainer.tick(2)
    assert not trainer.evolve() and len(trainer.history) == 2
    before = copy.deepcopy(trainer.snapshot())
    with pytest.raises(ValueError):
        trainer.set_generation_limit(1)
    assert trainer.generation_limit == 2 and trainer.snapshot() == before


@pytest.mark.parametrize('comparison_mode', [False, True])
def test_completed_workbench_can_extend_and_resume_without_losing_learning(monkeypatch, comparison_mode):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    bench = Workbench()
    bench.set_generations(2)
    if comparison_mode:
        bench.start_comparison()
    else:
        bench.set_running(True)
    for _ in range(20):
        bench.advance()
        if not bench.running:
            break
    completed = bench.snapshot()
    assert completed['complete'] and not completed['running']
    assert completed['generation_limit'] == completed['completed_generations'] == 2
    trainers = bench.comparison.trainers if comparison_mode else {'single': bench.trainer}
    saved = {key: (trainer, trainer.weights.copy(), copy.deepcopy(trainer.history)) for key, trainer in trainers.items()}
    extended = bench.set_generations(4)
    assert not extended['running'] and not extended['complete']
    assert extended['completed_generations'] == 2 and extended['generation_limit'] == 4
    for key, trainer in trainers.items():
        assert trainer is saved[key][0] and trainer.history == saved[key][2]
        np.testing.assert_array_equal(trainer.weights, saved[key][1])
    frozen = bench.snapshot()
    bench.advance()
    assert bench.snapshot() == frozen
    assert bench.set_running(True)['running']
    for _ in range(20):
        bench.advance()
        if not bench.running:
            break
    assert bench.snapshot()['complete'] and bench.snapshot()['completed_generations'] == 4
    assert all(trainer.history[:2] == saved[key][2] for key, trainer in trainers.items())


@pytest.mark.parametrize('comparison_mode', [False, True])
def test_seed_change_resets_all_learning_but_keeps_track_network_and_budget(monkeypatch, comparison_mode):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    bench = Workbench()
    bench.set_generations(250)
    bench.select_circuit('silverstone')
    bench.select_architecture('deep')
    if comparison_mode:
        bench.start_comparison()
    else:
        bench.set_running(True)
    for _ in range(3):
        bench.advance()
    before = bench.snapshot()
    assert bench.trainer.weights.any() and bench.trainer.history
    state = bench.set_seed(123)
    assert not state['running'] and not state['complete']
    assert state['seed'] == 123 and state['generation_limit'] == 250 and state['completed_generations'] == 0
    assert state['network']['id'] == 'deep' and state['track']['circuit_id'] == 'silverstone'
    assert state['track']['route'] == before['track']['route']
    assert state['revision'] > before['revision'] and state['sequence'] > before['sequence']
    assert state['generation'] == state['step_count'] == 0 and not state['history']
    trainers = bench.comparison.trainers.values() if comparison_mode else [bench.trainer]
    for trainer in trainers:
        assert trainer.seed == 123 and trainer.generation_limit == 250
        assert not trainer.weights.any() and trainer.best_weights is None and trainer.history == []
    assert (state['comparison'] is not None) == comparison_mode
    if comparison_mode:
        assert state['comparison']['seed'] == 123 and state['comparison']['status'] == 'paused'


@pytest.mark.parametrize('operation', ['reset', 'circuit', 'replace', 'architecture', 'comparison'])
def test_seed_and_budget_persist_when_changing_training_mode_or_track(operation):
    bench = Workbench()
    bench.set_generations(500)
    bench.set_seed(0)
    if operation == 'reset':
        state = bench.reset()
    elif operation == 'circuit':
        state = bench.select_circuit('interlagos')
    elif operation == 'replace':
        state = bench.replace_track(get_circuit('spa'))
    elif operation == 'architecture':
        state = bench.select_architecture('wide')
    else:
        bench.start_comparison()
        state = bench.start_comparison()
        assert state['comparison']['seed'] == 0 and state['comparison']['generation_limit'] == 500
        assert all(trainer.seed == 0 and trainer.generation_limit == 500 for trainer in bench.comparison.trainers.values())
    assert state['seed'] == bench.trainer.seed == 0
    assert state['generation_limit'] == bench.trainer.generation_limit == 500


def test_seed_repeats_mutations_after_reset_and_a_different_seed_changes_them(monkeypatch):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    bench = Workbench()
    bench.set_seed(4294967295)
    bench.select_architecture('wide')
    bench.trainer.tick(2)
    assert bench.trainer.evolve()
    expected = bench.trainer.weights.copy()
    assert expected.any()
    bench.reset()
    bench.trainer.tick(2)
    assert bench.trainer.evolve()
    np.testing.assert_array_equal(bench.trainer.weights, expected)
    bench.set_seed(0)
    bench.trainer.tick(2)
    assert bench.trainer.evolve()
    assert not np.array_equal(bench.trainer.weights, expected)


def test_comparison_seed_reproduces_independent_mutations_for_all_networks(monkeypatch):
    monkeypatch.setattr(engine_module, 'MAX_STEPS', 2)
    bench = Workbench()
    bench.set_seed(19)
    bench.set_generations(300)
    bench.start_comparison()
    bench.advance()
    bench.advance()
    for identifier, trainer in bench.comparison.trainers.items():
        reference = Trainer(bench.track, seed=19, architecture=identifier, generation_limit=300)
        reference.tick(2)
        assert reference.evolve()
        assert trainer.generation == 1 and trainer.weights.any()
        np.testing.assert_array_equal(trainer.weights, reference.weights)


def test_submitting_the_current_seed_preserves_the_active_run():
    bench = Workbench()
    bench.set_running(True)
    bench.advance()
    before = bench.snapshot()
    trainer = bench.trainer
    assert bench.set_seed(42) == before
    assert bench.trainer is trainer
