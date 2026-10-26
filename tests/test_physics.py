import numpy as np
import pytest

from racer.engine import OUTPUTS, PARAMETERS, Trainer
from racer.tracks import Track


def roomy_track(size=640, wall=None):
    mask = np.ones((size, size), dtype=bool)
    if wall is not None:
        mask[:, wall] = False
    route = size * np.array([[.5, .5], [.85, .5], [.85, .85], [.5, .85]])
    return Track(mask, np.full((size, size, 3), 255, dtype=np.uint8), route, route[0], 0, "Physics")


def fixed_policy(steering, target_speed=.7, brake=0):
    weights = np.zeros(PARAMETERS)
    weights[-OUTPUTS:] = np.arctanh([steering, target_speed, brake])
    return weights


def driver(speed_fraction, steering=.9, brake=0, size=640):
    trainer = Trainer(roomy_track(size), 16)
    trainer.weights[:] = fixed_policy(steering, target_speed=min(speed_fraction, float(np.tanh(8))), brake=brake)
    trainer.cars.speed[:] = trainer.max_speed * speed_fraction
    return trainer


def measured_turn(trainer):
    before = np.array([trainer.cars.x[0], trainer.cars.y[0]])
    heading = trainer.cars.heading[0]
    trainer.tick()
    after = np.array([trainer.cars.x[0], trainer.cars.y[0]])
    yaw = np.arctan2(np.sin(trainer.cars.heading[0] - heading), np.cos(trainer.cars.heading[0] - heading))
    return float(yaw), float(np.linalg.norm(after - before))


def test_fast_car_has_a_wider_turning_radius_for_the_same_steering():
    slow, fast = driver(.25), driver(1)
    slow_yaw, slow_distance = measured_turn(slow)
    fast_yaw, fast_distance = measured_turn(fast)
    assert slow.cars.alive.all() and fast.cars.alive.all()
    slow_radius = slow_distance / abs(slow_yaw)
    fast_radius = fast_distance / abs(fast_yaw)
    assert fast_radius > slow_radius * 2
    assert slow.cars.speed[0] == pytest.approx(slow.max_speed * .25)
    assert fast.cars.speed[0] == pytest.approx(fast.max_speed)


@pytest.mark.parametrize("steering", [-.99, -.7, .7, .99])
def test_high_speed_actual_motion_respects_lateral_grip(steering):
    trainer = driver(1, steering=steering)
    yaw, distance = measured_turn(trainer)
    #speed times yaw is the lateral acceleration for this physics step
    lateral_acceleration = trainer.cars.speed[0] * abs(yaw)
    assert lateral_acceleration <= trainer.max_lateral_acceleration * (1 + 1e-10)
    assert lateral_acceleration >= trainer.max_lateral_acceleration * .99
    assert distance > 0
    assert np.sign(yaw) == np.sign(steering)


@pytest.mark.parametrize("steering", [-.95, .95])
def test_steering_does_not_spin_a_stationary_car(steering):
    trainer = driver(0, steering=steering)
    trainer.weights[:] = fixed_policy(steering, target_speed=0)
    position = np.stack((trainer.cars.x, trainer.cars.y), axis=1)
    trainer.tick(12)
    np.testing.assert_array_equal(np.stack((trainer.cars.x, trainer.cars.y), axis=1), position)
    np.testing.assert_array_equal(trainer.cars.heading, np.zeros(trainer.population))
    assert trainer.cars.alive.all()


def test_left_and_right_steering_follow_mirrored_paths():
    trainer = driver(.7)
    trainer.weights[:8] = fixed_policy(-.8)
    trainer.weights[8:] = fixed_policy(.8)
    center_y = trainer.cars.y[0]
    trainer.tick(40)
    assert trainer.cars.alive.all()
    np.testing.assert_allclose(trainer.cars.x[:8], trainer.cars.x[8:], atol=1e-10)
    np.testing.assert_allclose(trainer.cars.y[:8] - center_y, center_y - trainer.cars.y[8:], atol=1e-10)
    np.testing.assert_allclose(trainer.cars.heading[:8], -trainer.cars.heading[8:], atol=1e-10)
    np.testing.assert_allclose(trainer.cars.speed[:8], trainer.cars.speed[8:])


def test_braking_tightens_the_turn_in_the_same_physics_step():
    coasting, braking = driver(1), driver(1, brake=.8)
    coast_yaw, coast_distance = measured_turn(coasting)
    brake_yaw, brake_distance = measured_turn(braking)
    assert braking.cars.speed[0] < coasting.cars.speed[0]
    assert brake_distance < coast_distance
    assert brake_distance / abs(brake_yaw) < .9 * coast_distance / abs(coast_yaw)


def test_braking_through_an_approach_increases_curvature():
    coasting, braking = driver(1), driver(1, brake=.5)
    coast_curvature, brake_curvature = [], []
    for _ in range(10):
        coast_yaw, coast_distance = measured_turn(coasting)
        brake_yaw, brake_distance = measured_turn(braking)
        coast_curvature.append(abs(coast_yaw) / coast_distance)
        brake_curvature.append(abs(brake_yaw) / brake_distance)
    assert braking.cars.alive.all() and coasting.cars.alive.all()
    assert np.all(np.diff(brake_curvature) > 0)
    assert brake_curvature[-1] > 2 * coast_curvature[-1]
    np.testing.assert_allclose(coast_curvature, coast_curvature[0], rtol=1e-10)
    assert braking.cars.heading[0] > coasting.cars.heading[0]


@pytest.mark.parametrize("size", [320, 640, 960])
def test_turning_geometry_scales_with_the_track(size):
    reference, scaled = driver(1, size=640), driver(1, size=size)
    reference_yaw, reference_distance = measured_turn(reference)
    scaled_yaw, scaled_distance = measured_turn(scaled)
    assert scaled_yaw == pytest.approx(reference_yaw, rel=1e-10)
    assert scaled_distance / scaled_yaw / scaled.scale == pytest.approx(
        reference_distance / reference_yaw / reference.scale, rel=2e-4,
    )
    assert scaled.max_lateral_acceleration / scaled.scale == pytest.approx(
        reference.max_lateral_acceleration / reference.scale,
    )


def test_turning_car_body_cannot_cross_a_one_pixel_wall_at_full_speed():
    trainer = Trainer(roomy_track(wall=333), 16)
    trainer.weights[:] = fixed_policy(.9, target_speed=.9)
    trainer.cars.speed[:] = trainer.max_speed
    trainer.tick()
    assert trainer.cars.crashed.all()
    assert not trainer.cars.alive.any()
    assert np.all(trainer.cars.speed == 0)
    assert trainer._on_road(trainer.cars.x, trainer.cars.y, trainer.cars.heading).all()
    front_extent = (np.cos(trainer.cars.heading) * trainer.car_length
                    + np.abs(np.sin(trainer.cars.heading)) * trainer.car_width) / 2
    assert np.all(trainer.cars.x + front_extent < 333)
