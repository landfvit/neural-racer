from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from racer.engine import Trainer
from racer.tracks import Track, VehicleProfile


def road(profile):
    route = np.array([[30., 40.], [170., 40.], [170., 170.], [30., 170.]])
    return Track(np.ones((200, 200), dtype=bool), np.full((200, 200, 3), 255, dtype=np.uint8),
                 route, route[0], 0, "Profile test", vehicle=profile)


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), True])
def test_invalid_geographic_scale_is_rejected(scale):
    with pytest.raises(ValueError):
        VehicleProfile.scaled(scale)


def test_vehicle_profile_is_immutable_and_rejects_invalid_geometry():
    profile = VehicleProfile.scaled(.5)
    with pytest.raises(FrozenInstanceError):
        profile.car_width = 100
    for change in ({"car_width": profile.car_length * 2}, {"wheelbase": profile.car_length * 2},
                   {"max_steer_angle": np.pi / 2}, {"gate_half_width": profile.car_width / 2},
                   {"drag": profile.acceleration}):
        with pytest.raises(ValueError):
            replace(profile, **change)


def test_map_scale_preserves_normalized_speed_and_cornering():
    small, large = [Trainer(road(VehicleProfile.scaled(scale)), 16) for scale in (.5, 1.)]
    assert small.max_speed * 2 == large.max_speed
    assert small.car_length * 2 == large.car_length
    assert small.gate_spacing == pytest.approx(large.gate_spacing / 2, rel=.02)
    small_turn = small.turn_rate(small.max_speed * .8, .6)
    large_turn = large.turn_rate(large.max_speed * .8, .6)
    assert small_turn == pytest.approx(large_turn)
    assert small.max_speed / small_turn * 2 == pytest.approx(large.max_speed / large_turn)


def test_one_motion_can_pass_multiple_ordered_small_gates_without_farming():
    profile = replace(VehicleProfile.scaled(1.), gate_spacing=1.)
    trainer = Trainer(road(profile), 16)
    before = np.repeat(trainer.gates[:1], 16, axis=0)
    after = before + trainer.normals[1] * 9.2
    active = np.arange(16) == 0
    trainer._advance_progress(before, after, active)
    assert trainer._passed[0] == 9
    assert not trainer._passed[1:].any()
    assert not trainer.cars.laps.any()
    for _ in range(3):
        trainer._advance_progress(before, after, active)
        trainer._advance_progress(after, before, active)
    assert trainer._passed[0] == 9
    assert not trainer.cars.laps.any()
