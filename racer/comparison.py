"""Equal generation budgets for independent network training runs."""

from __future__ import annotations

from .engine import MAX_GENERATIONS, Trainer, validate_generation_limit
from .networks import architecture_catalog


class Comparison:
    def __init__(self, track, seed=42, generation_limit=MAX_GENERATIONS):
        self.generation_limit = validate_generation_limit(generation_limit)
        self.population, self.seed = 64, seed
        self.trainers = {
            item["id"]: Trainer(track, population=self.population, seed=self.seed, architecture=item["id"],
                                generation_limit=self.generation_limit)
            for item in architecture_catalog()
        }

    @property
    def complete(self):
        return all(len(trainer.history) >= self.generation_limit for trainer in self.trainers.values())

    def set_generation_limit(self, generation_limit):
        generation_limit = validate_generation_limit(generation_limit)
        minimum = max(len(trainer.history) if trainer.done else trainer.generation + 1
                      for trainer in self.trainers.values())
        if generation_limit < minimum:
            raise ValueError(f"Generations must be at least {minimum} for this comparison. Reset to use a smaller budget.")
        for trainer in self.trainers.values():
            trainer.set_generation_limit(generation_limit)
        self.generation_limit = generation_limit

    def advance(self, steps):
        if self.complete:
            return
        #all networks finish the same generation before any starts the next
        if all(trainer.done for trainer in self.trainers.values()):
            for trainer in self.trainers.values():
                trainer.evolve()
        else:
            for trainer in self.trainers.values():
                if not trainer.done:
                    trainer.tick(steps)

    def snapshot(self, running):
        entries = []
        for identifier, trainer in self.trainers.items():
            history, best_time, first_lap = [], None, None
            for result in trainer.history:
                lap_time = result["lap_time"]
                if lap_time is not None:
                    best_time = lap_time if best_time is None else min(best_time, lap_time)
                    if first_lap is None:
                        first_lap = result["generation"]
                history.append({"generation": result["generation"], "lap_time": lap_time, "best_time": best_time})
            #a completed lap is valid immediately, even while other cars are driving
            if first_lap is None and trainer.best_ever_time is not None:
                first_lap = trainer.generation
            entries.append({
                "id": identifier, "name": trainer.architecture.name, "generation": trainer.generation,
                "completed_generations": len(trainer.history), "best_time": trainer.best_ever_time,
                "first_lap_generation": first_lap, "history": history,
                "leader": trainer.car_snapshot(trainer.best_index),
            })
        return {
            "status": "complete" if self.complete else "running" if running else "paused",
            "generation_limit": self.generation_limit, "population": self.population, "seed": self.seed,
            "completed_generations": min(len(trainer.history) for trainer in self.trainers.values()),
            "entries": entries,
        }
