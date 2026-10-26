import copy
import json

import numpy as np
import pytest

import racer.comparison as comparison_module
from racer.circuits import get_circuit
from racer.comparison import Comparison
from racer.engine import SIMULATION_HZ, Trainer
from racer.networks import architecture_catalog, get_architecture
from racer.service import Workbench


IDS = tuple(item['id'] for item in architecture_catalog())


class ShortTrainer:
    """Deterministic unequal episode lengths for testing the generation barrier."""

    def __init__(self, track, population=64, seed=42, architecture='compact'):
        self.track, self.population, self.seed = track, population, seed
        self.architecture = get_architecture(architecture)
        self.generation, self.step_count = 0, 0
        self.history, self.best_ever_time = [], None
        self.done, self.evolutions = False, 0
        self.calls = []
        self.duration = IDS.index(self.architecture.id) + 1

    @property
    def best_index(self):
        return 0

    def car_snapshot(self, index):
        return {'index': index, 'alive': not self.done, 'x': float(self.step_count), 'y': 0.0}

    def tick(self, steps):
        assert not self.done, 'finished episodes must wait without being ticked'
        self.calls.append((self.generation, steps))
        self.step_count += 1
        if self.step_count >= self.duration:
            self.done = True
            lap_time = None if self.generation < self.duration else float(50 - self.generation / 10 - self.duration)
            self.best_ever_time = lap_time if lap_time is not None else self.best_ever_time
            self.history.append({'generation': self.generation, 'lap_time': lap_time})

    def evolve(self):
        assert self.done
        assert self.generation < 99, 'there must not be a generation 100'
        self.evolutions += 1
        self.generation += 1
        self.done, self.step_count = False, 0
        return True


@pytest.fixture
def short_comparison(monkeypatch):
    monkeypatch.setattr(comparison_module, 'Trainer', ShortTrainer)
    return Comparison(get_circuit('monza'))


def test_comparison_uses_fresh_independent_populations_with_equal_starting_budgets():
    comparison = Comparison(get_circuit('monaco'))
    assert tuple(comparison.trainers) == IDS
    assert comparison.population == 64 and comparison.seed == 42 and comparison.generation_limit == 100
    for identifier, trainer in comparison.trainers.items():
        independent = Trainer(trainer.track, population=64, seed=42, architecture=identifier)
        assert trainer.track is comparison.trainers['compact'].track
        assert trainer.generation == trainer.step_count == 0
        assert trainer.history == [] and trainer.best_weights is None and trainer.best_ever_time is None
        assert not trainer.weights.any()
        np.testing.assert_array_equal(trainer.weights, independent.weights)
        assert not np.shares_memory(trainer.weights, independent.weights)
    state = comparison.snapshot(True)
    assert state['status'] == 'running' and state['completed_generations'] == 0
    assert all(entry['best_time'] is None and entry['first_lap_generation'] is None for entry in state['entries'])
    json.dumps(state, allow_nan=False)


def test_interleaving_matches_independent_real_simulation_and_evolution():
    comparison = Comparison(get_circuit('monza'))
    independent = {identifier: Trainer(comparison.trainers[identifier].track, population=64, seed=42,
                                      architecture=identifier) for identifier in IDS}
    while not all(trainer.done for trainer in comparison.trainers.values()):
        comparison.advance(128)
    for identifier, trainer in comparison.trainers.items():
        reference = independent[identifier]
        while not reference.done:
            reference.tick(128)
        assert trainer.snapshot() == reference.snapshot()
        np.testing.assert_array_equal(trainer.weights, reference.weights)
        assert len(trainer.history) == 1 and trainer.generation == 0
    comparison.advance(128)
    for identifier, trainer in comparison.trainers.items():
        reference = independent[identifier]
        assert reference.evolve()
        assert trainer.generation == reference.generation == 1
        np.testing.assert_array_equal(trainer.weights, reference.weights)
        assert trainer.snapshot() == reference.snapshot()
    for _ in range(4):
        comparison.advance(16)
        for trainer in independent.values():
            trainer.tick(16)
    assert all(comparison.trainers[identifier].snapshot() == reference.snapshot()
               for identifier, reference in independent.items())


def test_live_comparison_leaders_match_each_real_trainer_without_full_network_snapshots(monkeypatch):
    comparison = Comparison(get_circuit('monza'))
    while not all(trainer.done for trainer in comparison.trainers.values()):
        comparison.advance(128)
    comparison.advance(32)
    comparison.advance(32)
    expected = {
        identifier: trainer.snapshot()['cars'][trainer.best_index]
        for identifier, trainer in comparison.trainers.items()
    }
    assert all((leader['x'], leader['y']) != tuple(comparison.trainers[identifier].track.start)
               for identifier, leader in expected.items())

    def fail_full_snapshot(_trainer):
        raise AssertionError('comparison leaders must not serialize each full network')

    monkeypatch.setattr(Trainer, 'snapshot', fail_full_snapshot)
    state = comparison.snapshot(True)
    assert {entry['id']: entry['leader'] for entry in state['entries']} == expected
    json.dumps(state, allow_nan=False)


def test_comparison_leaders_are_detached_from_trainers_and_later_snapshots():
    comparison = Comparison(get_circuit('monza'))
    comparison.advance(16)
    before = comparison.snapshot(True)
    frozen = copy.deepcopy(before)
    comparison.advance(16)
    assert before == frozen
    before['entries'][0]['leader']['x'] = -1000.0
    before['entries'][0]['leader']['alive'] = False
    current = comparison.snapshot(True)
    trainer = comparison.trainers['compact']
    assert current['entries'][0]['leader'] == trainer.car_snapshot(trainer.best_index)
    assert current['entries'][0]['leader']['x'] != -1000.0
    assert current['entries'][0]['leader'] is not before['entries'][0]['leader']


def test_finished_episodes_keep_the_actual_inactive_leader_visible():
    comparison = Comparison(get_circuit('monza'))
    while not all(trainer.done for trainer in comparison.trainers.values()):
        comparison.advance(128)
    state = comparison.snapshot(True)
    for entry in state['entries']:
        trainer = comparison.trainers[entry['id']]
        expected = trainer.snapshot()['cars'][trainer.best_index]
        assert entry['leader'] == expected
        assert not entry['leader']['alive'] and entry['leader']['speed'] == 0
        assert entry['leader']['elapsed_time'] > 0
        assert entry['leader']['lap_time'] is None or entry['leader']['finished']


def test_finished_lap_remains_the_leader_while_other_cars_are_active():
    comparison = Comparison(get_circuit('monza'))
    trainer = comparison.trainers['deep']
    index = 3
    trainer.cars.alive[index] = False
    trainer.cars.finished[index] = True
    trainer.cars.laps[index] = 1
    trainer.cars.progress[index] = trainer.route_length
    trainer.cars.elapsed_steps[index] = trainer.cars.finish_steps[index] = 500
    trainer._update_fitness()
    assert trainer.best_index == index and trainer.cars.alive.any() and not trainer.done
    entry = next(item for item in comparison.snapshot(True)['entries'] if item['id'] == 'deep')
    assert entry['leader'] == trainer.snapshot()['cars'][index]
    assert entry['leader']['finished'] and not entry['leader']['alive']
    assert entry['leader']['lap_time'] == entry['leader']['elapsed_time'] == 500 / SIMULATION_HZ


def test_finished_networks_wait_without_extra_steps_until_all_four_finish(short_comparison):
    comparison = short_comparison
    comparison.advance(16)
    compact = comparison.trainers['compact']
    assert compact.done and compact.generation == 0 and len(compact.calls) == 1
    for _ in range(3):
        comparison.advance(16)
        assert compact.done and compact.generation == 0 and len(compact.calls) == 1
    assert all(trainer.done for trainer in comparison.trainers.values())
    assert [len(trainer.history) for trainer in comparison.trainers.values()] == [1] * 4
    comparison.advance(16)
    assert all(trainer.generation == 1 and not trainer.done for trainer in comparison.trainers.values())
    assert all(trainer.step_count == 0 for trainer in comparison.trainers.values())


def test_comparison_stops_after_exactly_one_hundred_equal_generations(short_comparison):
    comparison = short_comparison
    for _ in range(1000):
        comparison.advance(16)
        if comparison.complete:
            break
    assert comparison.complete
    before = copy.deepcopy(comparison.snapshot(True))
    for _ in range(5):
        comparison.advance(128)
    assert comparison.snapshot(False) == before
    assert before['status'] == 'complete' and before['completed_generations'] == 100
    for trainer, entry in zip(comparison.trainers.values(), before['entries']):
        assert trainer.done and trainer.generation == 99 and trainer.evolutions == 99
        assert len(trainer.calls) == 100 * trainer.duration
        assert entry['generation'] == 99 and entry['completed_generations'] == 100
        assert [item['generation'] for item in entry['history']] == list(range(100))
        assert entry['first_lap_generation'] == trainer.duration
        assert entry['best_time'] == min(item['lap_time'] for item in entry['history'] if item['lap_time'] is not None)


def test_missing_laps_stay_null_and_records_do_not_increase(short_comparison):
    comparison = short_comparison
    comparison.advance(16)
    state = comparison.snapshot(False)
    assert state['status'] == 'paused' and state['completed_generations'] == 0
    compact = state['entries'][0]
    assert compact['history'] == [{'generation': 0, 'lap_time': None, 'best_time': None}]
    assert compact['best_time'] is None and compact['first_lap_generation'] is None
    trainer = comparison.trainers['compact']
    trainer.history = [
        {'generation': 0, 'lap_time': None},
        {'generation': 1, 'lap_time': 20.0},
        {'generation': 2, 'lap_time': 25.0},
        {'generation': 3, 'lap_time': None},
        {'generation': 4, 'lap_time': 18.0},
    ]
    trainer.best_ever_time, trainer.generation = 18.0, 4
    entry = comparison.snapshot(True)['entries'][0]
    assert entry['first_lap_generation'] == 1 and entry['best_time'] == 18.0
    assert [item['best_time'] for item in entry['history']] == [None, 20.0, 20.0, 20.0, 18.0]
    json.dumps(comparison.snapshot(True), allow_nan=False)


def test_live_first_lap_in_generation_zero_is_visible_before_episode_ends(short_comparison):
    comparison = short_comparison
    trainer = comparison.trainers['compact']
    trainer.best_ever_time = 14.5
    assert not trainer.done and trainer.history == []
    entry = comparison.snapshot(True)['entries'][0]
    assert entry['best_time'] == 14.5 and entry['first_lap_generation'] == 0
    assert entry['history'] == []


def test_selected_network_is_only_an_inspection_choice_during_comparison():
    bench = Workbench()
    bench.select_architecture('deep')
    started = bench.start_comparison()
    assert started['running'] and started['network']['id'] == 'deep'
    assert bench.trainer is bench.comparison.trainers['deep']
    bench.advance()
    saved_trainers = bench.comparison.trainers.copy()
    saved_history = {key: trainer.history.copy() for key, trainer in saved_trainers.items()}
    before = bench.snapshot()
    selected = bench.select_architecture('split')
    assert selected['revision'] == before['revision']
    assert selected['sequence'] > before['sequence'] and selected['running']
    assert selected['track'] == before['track'] and selected['comparison'] == before['comparison']
    assert bench.trainer is saved_trainers['split']
    assert all(bench.comparison.trainers[key] is trainer and trainer.history == saved_history[key]
               for key, trainer in saved_trainers.items())
    assert selected['network']['id'] == bench.model()['architecture_id'] == 'split'
    assert bench.model() == saved_trainers['split'].model()
    assert len(bench.model()['weights']) == get_architecture('split').parameters
    assert len(selected['comparison']['entries']) == 4
    for entry in selected['comparison']['entries']:
        trainer = saved_trainers[entry['id']]
        assert entry['leader'] == trainer.snapshot()['cars'][trainer.best_index]


def test_pausing_freezes_every_network_and_resuming_advances_all_of_them():
    bench = Workbench()
    bench.start_comparison()
    bench.advance()
    bench.set_running(False)
    before = bench.snapshot()
    steps = {key: trainer.step_count for key, trainer in bench.comparison.trainers.items()}
    for _ in range(3):
        bench.advance()
    assert bench.snapshot() == before
    assert before['comparison']['status'] == 'paused'
    bench.set_running(True)
    bench.advance()
    assert all(trainer.step_count == steps[key] + bench.speed for key, trainer in bench.comparison.trainers.items())
    assert bench.snapshot()['comparison']['status'] == 'running'


@pytest.mark.parametrize('operation', ['reset', 'circuit', 'replace'])
def test_reset_or_track_replacement_exits_comparison_with_selected_network(operation):
    bench = Workbench()
    bench.start_comparison()
    bench.select_architecture('split')
    bench.advance()
    previous = bench.snapshot()
    if operation == 'reset':
        state = bench.reset()
        assert state['track']['circuit_id'] == 'monza'
    elif operation == 'circuit':
        state = bench.select_circuit('monaco')
        assert state['track']['circuit_id'] == 'monaco'
    else:
        state = bench.replace_track(get_circuit('spa'))
        assert state['track']['circuit_id'] is None
    assert bench.comparison is None and state['comparison'] is None
    assert state['network']['id'] == 'split'
    assert not state['running'] and state['generation'] == state['step_count'] == 0
    assert state['history'] == [] and state['metrics']['best_ever_time'] is None
    assert state['revision'] > previous['revision'] and state['sequence'] > previous['sequence']


@pytest.mark.parametrize('choice', ['unknown', '../compact', None, 42])
def test_invalid_network_choice_preserves_active_comparison(choice):
    bench = Workbench()
    bench.start_comparison()
    bench.advance()
    before = bench.snapshot()
    trainers = bench.comparison.trainers.copy()
    with pytest.raises(ValueError):
        bench.select_architecture(choice)
    assert bench.snapshot() == before
    assert all(bench.comparison.trainers[key] is trainer for key, trainer in trainers.items())


def test_completed_comparison_cannot_resume_and_finished_network_can_still_be_inspected():
    bench = Workbench()
    bench.start_comparison()
    #finish the budget through the coordinator without substituting physics data in a public run
    for trainer in bench.comparison.trainers.values():
        trainer.history = [{'generation': generation, 'lap_time': None} for generation in range(100)]
        trainer.generation, trainer.done = 99, True
    bench.advance()
    completed = bench.snapshot()
    assert completed['complete'] and not completed['running']
    assert completed['comparison']['status'] == 'complete'
    with pytest.raises(ValueError, match='complete'):
        bench.set_running(True)
    assert bench.snapshot() == completed
    selected = bench.select_architecture('deep')
    assert selected['complete'] and not selected['running']
    assert selected['comparison'] == completed['comparison']
    assert selected['network']['id'] == 'deep'
    restarted = bench.start_comparison()
    assert restarted['running'] and not restarted['complete']
    assert restarted['network']['id'] == 'deep'
    assert restarted['comparison']['completed_generations'] == 0
    assert all(entry['history'] == [] and entry['best_time'] is None for entry in restarted['comparison']['entries'])


def test_failed_network_pauses_the_entire_comparison(monkeypatch):
    bench = Workbench()
    bench.start_comparison()

    def fail(_steps):
        raise RuntimeError('test worker failure')

    monkeypatch.setattr(bench.comparison.trainers['wide'], 'tick', fail)
    with pytest.raises(RuntimeError, match='test worker failure'):
        bench.advance()
    assert not bench.running and bench.error
    frozen = bench.snapshot()
    bench.advance()
    assert bench.snapshot() == frozen
    assert frozen['comparison']['status'] == 'paused'
