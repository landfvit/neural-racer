from __future__ import annotations

import io
import logging
import threading
import time
from uuid import uuid4

from PIL import Image

from .circuits import circuit_catalog, get_circuit
from .comparison import Comparison
from .engine import MAX_GENERATIONS, Trainer
from .networks import get_architecture


class Workbench:
    def __init__(self):
        self.lock = threading.RLock()
        self.track = get_circuit("monza")
        self.circuit_id = "monza"
        self.difficulty = next(item["difficulty"] for item in circuit_catalog() if item["id"] == "monza")
        self.seed, self.generation_limit = 42, MAX_GENERATIONS
        self.trainer = Trainer(self.track, seed=self.seed, generation_limit=self.generation_limit)
        self.comparison = None
        self.running, self.speed = False, 4
        self.revision, self.sequence = 1, 1
        self.instance = uuid4().hex
        self.error = None
        self._image = self._encode_image(self.track)
        self._stop = threading.Event()
        self._worker = None

    @staticmethod
    def _encode_image(track):
        buffer = io.BytesIO()
        Image.fromarray(track.image).save(buffer, format="PNG")
        return buffer.getvalue()

    def start(self):
        with self.lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(target=self._loop, name="car-training", daemon=True)
            self._worker.start()

    def stop(self):
        with self.lock:
            self._stop.set()
            worker = self._worker
        if worker is not None:
            worker.join(timeout=3)
        with self.lock:
            if self._worker is worker and (worker is None or not worker.is_alive()):
                self._worker = None

    def _loop(self):
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self.advance()
            except Exception:
                logging.exception("Training stopped unexpectedly")
                with self.lock:
                    self.running = False
                    self.error = "Training stopped. Reset the run and try again."
                    self.sequence += 1
            self._stop.wait(max(0.001, 1 / 30 - (time.monotonic() - started)))

    def advance(self):
        with self.lock:
            if not self.running:
                return
            try:
                if self.comparison is not None:
                    self.comparison.advance(self.speed)
                    if self.comparison.complete:
                        self.running = False
                elif self.trainer.done:
                    if not self.trainer.evolve():
                        self.running = False
                else:
                    self.trainer.tick(self.speed)
                    if self.trainer.complete:
                        self.running = False
            except Exception:
                #one failed trainer pauses the whole comparison under the same lock
                self.running = False
                self.error = "Training stopped. Reset the run and try again."
                self.sequence += 1
                raise
            self.sequence += 1

    def snapshot(self):
        with self.lock:
            payload = self.trainer.snapshot()
            comparison = self.comparison.snapshot(self.running) if self.comparison is not None else None
            return {
                **payload, "instance": self.instance, "revision": self.revision, "sequence": self.sequence,
                "running": self.running, "speed": self.speed, "error": self.error,
                "seed": self.seed, "generation_limit": self.generation_limit,
                "completed_generations": comparison["completed_generations"] if comparison is not None else len(self.trainer.history),
                "complete": self.comparison.complete if self.comparison is not None else self.trainer.complete,
                "comparison": comparison,
                "track": {"name": self.track.name, "width": self.track.width, "height": self.track.height,
                          "circuit_id": self.circuit_id, "difficulty": self.difficulty,
                          "image_url": f"/api/track/image?revision={self.revision}",
                          "route": self.track.route.tolist(), "start": list(map(float, self.track.start)),
                          "car_length": self.trainer.car_length, "car_width": self.trainer.car_width},
            }

    def image(self, revision):
        with self.lock:
            if revision is not None and revision != self.revision:
                raise ValueError("That track image has been replaced.")
            return self._image

    def replace_track(self, track, *, circuit_id=None, difficulty=None):
        #prepare everything before replacing a working run
        image = self._encode_image(track)
        with self.lock:
            trainer = Trainer(track, seed=self.seed, architecture=self.trainer.architecture.id,
                              generation_limit=self.generation_limit)
            self.track, self.trainer, self._image = track, trainer, image
            self.comparison = None
            self.circuit_id, self.difficulty = circuit_id, difficulty
            self.running, self.error = False, None
            self.revision += 1
            self.sequence += 1
            return self.snapshot()

    def select_circuit(self, circuit_id):
        track = get_circuit(circuit_id)
        details = next(item for item in circuit_catalog() if item["id"] == circuit_id)
        return self.replace_track(track, circuit_id=circuit_id, difficulty=details["difficulty"])

    def reset(self):
        with self.lock:
            self.trainer = Trainer(self.track, seed=self.seed, architecture=self.trainer.architecture.id,
                                   generation_limit=self.generation_limit)
            self.comparison = None
            self.running, self.error = False, None
            self.revision += 1
            self.sequence += 1
            return self.snapshot()

    def select_architecture(self, architecture_id):
        architecture = get_architecture(architecture_id)
        with self.lock:
            if self.comparison is not None:
                self.trainer = self.comparison.trainers[architecture.id]
                self.sequence += 1
                return self.snapshot()
            trainer = Trainer(self.track, seed=self.seed, architecture=architecture.id,
                              generation_limit=self.generation_limit)
            self.trainer = trainer
            self.running, self.error = False, None
            self.revision += 1
            self.sequence += 1
            return self.snapshot()

    def start_comparison(self):
        with self.lock:
            selected = self.trainer.architecture.id
            comparison = Comparison(self.track, seed=self.seed, generation_limit=self.generation_limit)
            self.comparison, self.trainer = comparison, comparison.trainers[selected]
            self.running, self.error = True, None
            self.revision += 1
            self.sequence += 1
            return self.snapshot()

    def set_running(self, running):
        with self.lock:
            if running and self.comparison is not None and self.comparison.complete:
                raise ValueError("This comparison is complete. Increase generations or reset to train again.")
            if running and self.comparison is None and self.trainer.complete:
                raise ValueError("This run is complete. Increase generations or reset to train again.")
            self.running, self.error = running, None
            self.sequence += 1
            return self.snapshot()

    def set_speed(self, speed):
        with self.lock:
            self.speed = speed
            self.sequence += 1
            return self.snapshot()

    def set_generations(self, generations):
        with self.lock:
            if self.comparison is not None:
                self.comparison.set_generation_limit(generations)
            else:
                self.trainer.set_generation_limit(generations)
            self.generation_limit = generations
            complete = self.comparison.complete if self.comparison is not None else self.trainer.complete
            if complete:
                self.running = False
            self.sequence += 1
            return self.snapshot()

    def set_seed(self, seed):
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Seed must be an integer from 0 to 4294967295.")
        with self.lock:
            if seed == self.seed:
                return self.snapshot()
            selected = self.trainer.architecture.id
            if self.comparison is not None:
                comparison = Comparison(self.track, seed=seed, generation_limit=self.generation_limit)
                trainer = comparison.trainers[selected]
            else:
                comparison = None
                trainer = Trainer(self.track, seed=seed, architecture=selected,
                                  generation_limit=self.generation_limit)
            self.seed, self.comparison, self.trainer = seed, comparison, trainer
            self.running, self.error = False, None
            self.revision += 1
            self.sequence += 1
            return self.snapshot()

    def model(self):
        with self.lock:
            return self.trainer.model()
