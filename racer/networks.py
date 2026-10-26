"""Small fixed network choices with explicit driving outputs."""

from __future__ import annotations

from dataclasses import dataclass


INPUT_NAMES = ("wall_left_60", "wall_left_30", "wall_forward", "wall_right_30", "wall_right_60")


@dataclass(frozen=True)
class Architecture:
    id: str
    name: str
    hidden: tuple[int, ...]
    outputs: tuple[str, ...]
    description: str

    @property
    def sizes(self):
        return (len(INPUT_NAMES), *self.hidden, len(self.outputs))

    @property
    def parameters(self):
        return sum((previous + 1) * current for previous, current in zip(self.sizes, self.sizes[1:]))


_ARCHITECTURES = (
    Architecture("compact", "Compact", (12,), ("steering", "target_speed", "brake"),
                 "One small hidden layer with steering, target speed, and brake outputs."),
    Architecture("wide", "Wide", (24,), ("steering", "target_speed", "brake"),
                 "A wider hidden layer with steering, target speed, and brake outputs."),
    Architecture("deep", "Deep", (16, 12), ("steering", "target_speed", "brake"),
                 "Two hidden layers with steering, target speed, and brake outputs."),
    Architecture("split", "Split steering", (20, 12), ("steering_left", "steering_right", "target_speed", "brake"),
                 "Two hidden layers with separate steering, target speed, and brake outputs."),
)


def get_architecture(identifier="compact"):
    if isinstance(identifier, Architecture):
        if identifier in _ARCHITECTURES:
            return identifier
    elif isinstance(identifier, str):
        for architecture in _ARCHITECTURES:
            if architecture.id == identifier:
                return architecture
    raise ValueError("Choose Compact, Wide, Deep, or Split steering.")


def architecture_catalog():
    return [{"id": item.id, "name": item.name, "hidden": list(item.hidden), "outputs": list(item.outputs),
             "sizes": list(item.sizes), "parameters": item.parameters, "description": item.description}
            for item in _ARCHITECTURES]
