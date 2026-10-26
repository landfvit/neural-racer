"""Incremental neuroevolution of cars that choose speed from wall rays."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .networks import INPUT_NAMES as NETWORK_INPUTS, get_architecture


INPUTS, HIDDEN, OUTPUTS = 5, 12, 3
PARAMETERS = INPUTS * HIDDEN + HIDDEN + HIDDEN * OUTPUTS + OUTPUTS
MAX_STEPS, MAX_GENERATIONS = 1400, 100
MAX_GENERATION_LIMIT = 10_000
SIMULATION_HZ = 30
STALL_STEPS, LAP_LIMIT = 150, 1
RAY_ANGLES = np.deg2rad([-60, -30, 0, 30, 60])
INPUT_NAMES = list(NETWORK_INPUTS)
OUTPUT_NAMES = ["steering", "target_speed", "brake"]


def validate_generation_limit(value):
    if type(value) is not int or not 1 <= value <= MAX_GENERATION_LIMIT:
        raise ValueError("Generations must be an integer from 1 to 10000.")
    return value


def _weights(value, architecture="compact"):
    specification = get_architecture(architecture)
    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Policy weights must be finite numbers between -8 and 8.") from exc
    if result.ndim == 1:
        result = result[None]
    if result.ndim != 2 or not len(result) or result.shape[1] != specification.parameters or not np.isfinite(result).all() or (np.abs(result) > 8).any():
        raise ValueError(f"Each policy requires {specification.parameters} finite weights between -8 and 8.")
    return result


def _layers(weights, architecture):
    cut = 0
    for previous, current in zip(architecture.sizes, architecture.sizes[1:]):
        end = cut + previous * current
        matrix = weights[:, cut:end].reshape(-1, previous, current)
        bias = weights[:, end:end + current]
        yield matrix, bias
        cut = end + current


def _forward(weights, observations, architecture="compact"):
    specification = get_architecture(architecture)
    weights = _weights(weights, specification)
    try:
        inputs = np.asarray(observations, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Observations must contain five finite values per policy.") from exc
    if inputs.shape != (len(weights), INPUTS) or not np.isfinite(inputs).all():
        raise ValueError("Observations must contain five finite values per policy.")
    activity = inputs
    hidden = []
    for matrix, bias in _layers(weights, specification):
        activity = np.tanh(np.einsum("ni,nij->nj", activity, matrix) + bias)
        hidden.append(activity)
    return hidden[:-1], hidden[-1]


def policy(weights, observations, architecture="compact"):
    return _forward(weights, observations, architecture)[1]


def network(weights, architecture="compact"):
    specification = get_architecture(architecture)
    vector = _weights(weights, specification)[:1]
    layers = [{"weights": matrix[0].tolist(), "bias": bias[0].tolist()}
              for matrix, bias in _layers(vector, specification)]
    return {
        "id": specification.id, "name": specification.name, "sizes": list(specification.sizes),
        "parameters": specification.parameters, "inputs": INPUT_NAMES.copy(), "outputs": list(specification.outputs),
        "layers": layers,
        "input_hidden": layers[0]["weights"], "hidden_bias": layers[0]["bias"],
        "hidden_output": layers[-1]["weights"], "output_bias": layers[-1]["bias"],
    }


@dataclass
class Cars:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    alive: np.ndarray
    progress: np.ndarray
    laps: np.ndarray
    fitness: np.ndarray
    crashed: np.ndarray
    wrong_way: np.ndarray
    elapsed_steps: np.ndarray
    finish_steps: np.ndarray
    finished: np.ndarray
    throttle: np.ndarray
    brake: np.ndarray
    steering: np.ndarray
    target_speed: np.ndarray


class Trainer:
    def __init__(self, track, population=64, seed=42, architecture="compact", generation_limit=MAX_GENERATIONS):
        if type(population) is not int or not 16 <= population <= 128:
            raise ValueError("Population must be an integer from 16 to 128.")
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer from 0 to 4294967295.")
        self.generation_limit = validate_generation_limit(generation_limit)
        self.track, self.population, self.seed = track, population, seed
        self.architecture = get_architecture(architecture)
        self.rng = np.random.default_rng(seed)
        self.scale = float(min(track.width, track.height))
        self.car_length, self.car_width = self.scale * 0.035, self.scale * 0.018
        self.max_speed, self.ray_range = self.scale * 0.008, self.scale * 0.30
        self.wheelbase = self.car_length * 0.65
        self.max_steer_angle = 0.38
        self.max_lateral_acceleration = self.scale * 0.00036
        self.acceleration, self.braking, self.drag = self.scale * np.array([0.00032, 0.00080, 0.000025])
        self.route_gate_spacing, self.gate_half_width = self.scale * 0.04, self.scale * 0.13
        self.progress_threshold = self.scale * 0.005
        if track.vehicle is not None:
            profile = track.vehicle
            for name in ("car_length", "car_width", "wheelbase", "max_speed", "ray_range", "max_steer_angle",
                         "max_lateral_acceleration", "acceleration", "braking", "drag", "gate_half_width", "progress_threshold"):
                setattr(self, name, getattr(profile, name))
            self.route_gate_spacing = profile.gate_spacing
        self.generation, self.history = 0, []
        #fresh runs begin with no learned connections or output bias, then mutations explore
        self.weights = np.zeros((population, self.architecture.parameters), dtype=float)
        self.best_weights, self.best_fitness, self.best_generation = None, float("-inf"), None
        self.best_ever_time = None
        self._prepare_route()
        self.start_generation()
        if not self._on_road(self.cars.x, self.cars.y, self.cars.heading).all():
            raise ValueError("The starting point needs enough road clearance for the car body.")

    def _prepare_route(self):
        route = np.asarray(self.track.route, dtype=float)
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 4 or not np.isfinite(route).all():
            raise ValueError("The track requires an ordered closed route with at least four points.")
        delta = np.roll(route, -1, axis=0) - route
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths <= 1e-6):
            raise ValueError("Route points must be distinct from their neighbors.")
        self.route_length = float(lengths.sum())
        count = max(12, int(np.ceil(self.route_length / self.route_gate_spacing)))
        if count > 3000:
            raise ValueError("The route is too long for this image.")
        self.gate_spacing = self.route_length / count
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        distances = np.arange(count) * self.gate_spacing
        segments = np.searchsorted(cumulative[1:], distances, side="right")
        fraction = (distances - cumulative[segments]) / lengths[segments]
        self.gates = route[segments] + delta[segments] * fraction[:, None]
        tangent = np.roll(self.gates, -1, axis=0) - np.roll(self.gates, 1, axis=0)
        self.normals = tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)

    def start_generation(self):
        size = self.population
        self.cars = Cars(
            np.full(size, float(self.track.start[0])), np.full(size, float(self.track.start[1])),
            np.full(size, float(self.track.heading)), np.zeros(size), np.ones(size, dtype=bool),
            np.zeros(size), np.zeros(size, dtype=int), np.zeros(size), np.zeros(size, dtype=bool), np.zeros(size, dtype=bool),
            np.zeros(size, dtype=int), np.full(size, -1, dtype=int), np.zeros(size, dtype=bool),
            np.zeros(size), np.zeros(size), np.zeros(size), np.zeros(size),
        )
        self._passed = np.zeros(size, dtype=int)
        self._last_gain = np.zeros(size, dtype=int)
        self._last_progress = np.zeros(size)
        self.step_count, self.done = 0, False

    def _mask(self, points):
        x, y = np.floor(points[..., 0]).astype(int), np.floor(points[..., 1]).astype(int)
        valid = (x >= 0) & (y >= 0) & (x < self.track.width) & (y < self.track.height)
        return valid & self.track.mask[np.clip(y, 0, self.track.height - 1), np.clip(x, 0, self.track.width - 1)]

    def _on_road(self, x, y, heading):
        forward = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
        side = np.stack((-forward[:, 1], forward[:, 0]), axis=-1)
        center = np.stack((x, y), axis=-1)
        points = [center]
        for front, lateral in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            points.append(center + forward * (front * self.car_length / 2) + side * (lateral * self.car_width / 2))
        return self._mask(np.stack(points, axis=1)).all(axis=1)

    def sensors(self):
        heading = self.cars.heading[:, None] + RAY_ANGLES
        directions = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
        #sample narrow scaled roads finely enough to detect a single background pixel
        samples = max(32, int(np.ceil(self.ray_range / 0.70))) if self.track.vehicle is not None else 32
        fractions = np.linspace(1 / samples, 1, samples)
        positions = np.stack((self.cars.x, self.cars.y), axis=-1)
        points = positions[:, None, None, :] + directions[:, :, None, :] * (fractions * self.ray_range)[None, None, :, None]
        road = self._mask(points)
        first = np.argmax(~road, axis=-1)
        values = np.where(road.all(axis=-1), 1.0, fractions[first])
        endpoints = positions[:, None, :] + directions * (values * self.ray_range)[..., None]
        return {"values": values, "endpoints": endpoints}

    def _advance_progress(self, before, after, active):
        crossing = active.copy()
        #small real-world circuits may cross more than one ordered gate per step
        for _ in range(int(np.ceil(self.max_speed / self.gate_spacing)) + 1):
            next_index = (self._passed + 1) % len(self.gates)
            gate, normal = self.gates[next_index], self.normals[next_index]
            previous_side = np.sum((before - gate) * normal, axis=1)
            current_side = np.sum((after - gate) * normal, axis=1)
            lateral = np.abs((after[:, 0] - gate[:, 0]) * normal[:, 1] - (after[:, 1] - gate[:, 1]) * normal[:, 0])
            passed = crossing & (previous_side < 0) & (current_side >= 0) & (lateral <= self.gate_half_width)
            self._passed += passed
            crossing &= passed & (self._passed < len(self.gates) * LAP_LIMIT)
            if not crossing.any():
                break
        self.cars.laps = self._passed // len(self.gates)
        next_index = (self._passed + 1) % len(self.gates)
        remaining = np.sum((after - self.gates[next_index]) * self.normals[next_index], axis=1)
        fraction = np.clip(1 + remaining / self.gate_spacing, 0, 1)
        candidate = (self._passed + fraction) * self.gate_spacing
        distance = np.linalg.norm(after - before, axis=1)
        progress = np.minimum(np.maximum(self.cars.progress, candidate), self.cars.progress + distance * 1.25)
        progress = np.minimum(progress, self.route_length * LAP_LIMIT)
        gained = active & (progress > self._last_progress + self.progress_threshold)
        self._last_gain[gained] = self.step_count
        self._last_progress[gained] = progress[gained]
        self.cars.progress = np.where(active, progress, self.cars.progress)
        toward = np.sum((after - before) * self.normals[next_index], axis=1)
        self.cars.wrong_way = active & (toward < -distance * 0.15) & (distance > self.scale * 0.0001)

    def turn_rate(self, speed, steering):
        speed = np.maximum(np.asarray(speed, dtype=float), 0)
        curvature = np.tan(np.clip(steering, -1, 1) * self.max_steer_angle) / self.wheelbase
        #lateral grip limits curvature at high speed, so braking allows tighter turns
        grip = np.divide(self.max_lateral_acceleration, speed * speed,
                         out=np.full_like(speed, np.inf), where=speed > 0)
        return speed * np.clip(curvature, -grip, grip)

    def tick(self, steps=1):
        if type(steps) is not int or not 1 <= steps <= 128:
            raise ValueError("Advance between 1 and 128 physics steps at a time.")
        for _ in range(steps):
            if self.done:
                break
            active = self.cars.alive.copy()
            actions = policy(self.weights, self.sensors()["values"], self.architecture)
            if self.architecture.id == "split":
                steering = np.clip(actions[:, 1], 0, 1) - np.clip(actions[:, 0], 0, 1)
                speed_output, brake_output = actions[:, 2], actions[:, 3]
            else:
                steering = actions[:, 0]
                speed_output, brake_output = actions[:, 1], actions[:, 2]
            target_speed = np.clip(speed_output, 0, 1)
            target = target_speed * self.max_speed
            accelerate, decelerate, drag = self.acceleration, self.braking, self.drag
            #the internal speed regulator applies pedals without feeding speed to the network
            throttle = np.where(target > 0, np.clip((target - self.cars.speed + drag) / accelerate, 0, 1), 0)
            automatic_brake = np.clip((self.cars.speed - target - drag) / decelerate, 0, 1)
            brake = np.maximum(automatic_brake, np.clip(brake_output, 0, 1))
            throttle *= 1 - brake
            if self.track.vehicle is None:
                acceleration = self.scale * (0.00032 * throttle - 0.00080 * brake - 0.000025)
            else:
                acceleration = accelerate * throttle - decelerate * brake - drag
            speed = np.clip(self.cars.speed + acceleration, 0, self.max_speed)
            speed = np.where(active, speed, self.cars.speed)
            turn = self.turn_rate(speed, steering)
            before = np.stack((self.cars.x, self.cars.y), axis=-1)
            #substeps keep the whole body from crossing a thin wall
            sweep = speed + np.abs(turn) * np.hypot(self.car_length, self.car_width) / 2
            substeps = np.maximum(1, np.ceil(sweep / 0.70).astype(int))
            collided = np.zeros(self.population, dtype=bool)
            for part in range(int(substeps.max())):
                moving = active & ~collided & (part < substeps)
                yaw = self.cars.heading + turn / substeps
                yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
                x = self.cars.x + np.cos(yaw) * speed / substeps
                y = self.cars.y + np.sin(yaw) * speed / substeps
                road = self._on_road(x, y, yaw)
                collided |= moving & ~road
                accept = moving & road
                self.cars.x = np.where(accept, x, self.cars.x)
                self.cars.y = np.where(accept, y, self.cars.y)
                self.cars.heading = np.where(accept, yaw, self.cars.heading)
            self.step_count += 1
            self.cars.elapsed_steps += active
            self.cars.crashed |= collided
            self.cars.speed = np.where(active & ~collided, speed, 0)
            after = np.stack((self.cars.x, self.cars.y), axis=-1)
            self._advance_progress(before, after, active)
            completed = active & (self.cars.laps >= LAP_LIMIT)
            self.cars.finish_steps[completed] = self.cars.elapsed_steps[completed]
            self.cars.finished |= completed
            self._update_fitness()
            if completed.any():
                lap_time = self._best_time()
                self.best_ever_time = lap_time if self.best_ever_time is None else min(self.best_ever_time, lap_time)
                index = self.best_index
                if self.cars.fitness[index] > self.best_fitness:
                    self.best_fitness, self.best_generation = float(self.cars.fitness[index]), self.generation
                    self.best_weights = self.weights[index].copy()
            stalled = self.step_count - self._last_gain >= STALL_STEPS
            self.cars.alive &= ~collided & ~stalled & (self.cars.laps < LAP_LIMIT)
            self.cars.speed[~self.cars.alive] = 0
            self.cars.throttle = np.where(self.cars.alive, throttle, 0)
            self.cars.brake = np.where(self.cars.alive, brake, 0)
            self.cars.steering = np.where(self.cars.alive, steering, 0)
            self.cars.target_speed = np.where(self.cars.alive, target_speed, 0)
            if self.step_count >= MAX_STEPS or not self.cars.alive.any():
                self.cars.alive[:] = False
                self.cars.speed[:] = 0
                self.cars.throttle[:] = 0
                self.cars.brake[:] = 0
                self.cars.steering[:] = 0
                self.cars.target_speed[:] = 0
                self.done = True
                self._finish()
        return self.done

    def _update_fitness(self):
        #every completed lap beats any partial lap, then fewer steps win
        partial = np.minimum(1000 * self.cars.progress / self.route_length, 999.999) - self.cars.crashed * 2
        timed = 1000 + 100 * (1 - self.cars.finish_steps / MAX_STEPS)
        self.cars.fitness = np.where(self.cars.finished, timed, partial)

    def _best_time(self):
        times = self.cars.finish_steps[self.cars.finished]
        return float(times.min() / SIMULATION_HZ) if len(times) else None

    @property
    def best_index(self):
        return int(np.argmax(self.cars.fitness))

    def _finish(self):
        index = self.best_index
        score = float(self.cars.fitness[index])
        if score > self.best_fitness:
            self.best_fitness, self.best_generation = score, self.generation
            self.best_weights = self.weights[index].copy()
        lap_time = self._best_time()
        if lap_time is not None:
            self.best_ever_time = lap_time if self.best_ever_time is None else min(self.best_ever_time, lap_time)
        self.history.append({
            "generation": self.generation, "best_fitness": score, "mean_fitness": float(self.cars.fitness.mean()),
            "best_progress": float(self.cars.progress[index]), "best_laps": int(self.cars.laps[index]),
            "lap_time": lap_time,
        })

    def evolve(self):
        if not self.done:
            raise ValueError("Finish the current generation before evolving.")
        if self.complete:
            return False
        order = np.argsort(-self.cars.fitness, kind="stable")
        elite = max(2, self.population // 16)
        pool = max(4, self.population // 4)
        parents = self.rng.integers(0, pool, (self.population - elite, 2))
        mix = self.rng.uniform(0, 1, (len(parents), 1))
        ranked = self.weights[order]
        children = ranked[parents[:, 0]] * mix + ranked[parents[:, 1]] * (1 - mix)
        scales = self.rng.choice([0.035, 0.12, 0.30], (len(children), 1), p=[0.25, 0.6, 0.15])
        children += self.rng.normal(0, 1, children.shape) * scales * (self.rng.random(children.shape) < 0.4)
        self.weights = np.concatenate((ranked[:elite].copy(), np.clip(children, -8, 8)))
        self.generation += 1
        self.start_generation()
        return True

    @property
    def complete(self):
        return self.done and len(self.history) >= self.generation_limit

    @property
    def minimum_generation_limit(self):
        return len(self.history) if self.done else self.generation + 1

    def set_generation_limit(self, generation_limit):
        generation_limit = validate_generation_limit(generation_limit)
        if generation_limit < self.minimum_generation_limit:
            raise ValueError(f"Generations must be at least {self.minimum_generation_limit} for this run. Reset to use a smaller budget.")
        self.generation_limit = generation_limit

    def activity(self, index=None):
        index = self.best_index if index is None else index
        if type(index) is not int or not 0 <= index < self.population:
            raise ValueError("Choose a car from the population.")
        inputs = self.sensors()["values"][index:index + 1]
        hidden, outputs = _forward(self.weights[index:index + 1], inputs, self.architecture)
        return {"inputs": inputs[0].tolist(), "hidden": hidden[0][0].tolist(),
                "hidden_layers": [layer[0].tolist() for layer in hidden], "outputs": outputs[0].tolist()}

    def network(self, index=None):
        return network(self.weights[self.best_index if index is None else index], self.architecture)

    def car_snapshot(self, index):
        if type(index) is not int or not 0 <= index < self.population:
            raise ValueError("Choose a car from the population.")
        return {
            "index": index, "x": float(self.cars.x[index]), "y": float(self.cars.y[index]), "heading": float(self.cars.heading[index]),
            "speed": float(self.cars.speed[index]), "alive": bool(self.cars.alive[index]), "crashed": bool(self.cars.crashed[index]),
            "throttle": float(self.cars.throttle[index]), "brake": float(self.cars.brake[index]),
            "steering": float(self.cars.steering[index]),
            "target_speed": float(self.cars.target_speed[index]),
            "progress": float(self.cars.progress[index]), "laps": int(self.cars.laps[index]), "fitness": float(self.cars.fitness[index]),
            "wrong_way": bool(self.cars.wrong_way[index]),
            "elapsed_time": float(self.cars.elapsed_steps[index] / SIMULATION_HZ),
            "lap_time": float(self.cars.finish_steps[index] / SIMULATION_HZ) if self.cars.finished[index] else None,
            "finished": bool(self.cars.finished[index]),
        }

    def snapshot(self):
        index = self.best_index
        sensors = self.sensors()
        brain = self.activity(index)
        cars = [self.car_snapshot(i) for i in range(self.population)]
        return {
            "generation": self.generation, "step_count": self.step_count, "done": self.done, "best_index": index,
            "seed": self.seed, "generation_limit": self.generation_limit, "completed_generations": len(self.history),
            "cars": cars, "sensors": {"values": sensors["values"][index].tolist(), "endpoints": sensors["endpoints"][index].tolist()},
            "brain": brain, "network": self.network(index), "history": [dict(item) for item in self.history],
            "controls": {"steering": float(self.cars.steering[index]), "target_speed": float(self.cars.target_speed[index]),
                         "throttle": float(self.cars.throttle[index]), "brake": float(self.cars.brake[index])},
            "metrics": {"best_fitness": float(self.cars.fitness[index]), "mean_fitness": float(self.cars.fitness.mean()),
                        "best_progress": float(self.cars.progress[index]), "best_laps": int(self.cars.laps[index]),
                        "best_time": self._best_time(), "best_ever_time": self.best_ever_time,
                        "best_ever_fitness": None if self.best_weights is None else self.best_fitness},
        }

    def model(self):
        weights = self.best_weights if self.best_weights is not None else self.weights[self.best_index]
        hidden = list(self.architecture.hidden)
        physics = {"model": "bicycle_grip_v1", "wheelbase_ratio": self.wheelbase / self.scale,
                   "max_steer_angle": self.max_steer_angle,
                   "lateral_acceleration_ratio": self.max_lateral_acceleration / self.scale}
        if self.track.vehicle is not None:
            physics["vehicle_profile"] = asdict(self.track.vehicle)
        return {"format": "neural-racer-policy", "version": 2,
                "architecture_id": self.architecture.id,
                "architecture": {"inputs": INPUTS, "hidden": hidden[0] if len(hidden) == 1 else hidden,
                                 "outputs": len(self.architecture.outputs), "activation": "tanh"},
                "weights": weights.tolist(), "generation": self.best_generation if self.best_weights is not None else self.generation,
                "seed": self.seed, "inputs": INPUT_NAMES.copy(), "outputs": list(self.architecture.outputs),
                "objective": "minimum_lap_time", "simulation_hz": SIMULATION_HZ, "lap_limit": LAP_LIMIT,
                "physics": physics,
                "lap_time": self.best_ever_time}

    def load_model(self, model):
        from .models import validate_model
        validated = validate_model(model)
        self.architecture = get_architecture(validated["architecture_id"])
        self.weights = np.repeat(np.asarray(validated["weights"])[None], self.population, axis=0)
        self.generation, self.history = 0, []
        self.best_weights, self.best_fitness, self.best_generation = None, float("-inf"), None
        self.best_ever_time = None
        self.start_generation()
