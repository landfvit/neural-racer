# Neural Racer

A local Python simulator where neural networks learn to drive faster laps. Watch training, inspect the networks, compare architectures, or upload a track image.

## Run locally

Use Python 3.14 and uv. Open the project folder in Terminal and run these commands.

```text
uv sync --locked
uv run uvicorn app:app --host 127.0.0.1 --port 8002
```

Open [localhost:8002](http://127.0.0.1:8002/). Closing the Terminal session stops the server. Run tests with this command.

```text
uv run --extra dev pytest -q
```

## Use the workbench

Choose a circuit and network, then click **Train**. Translucent ghost cars show the population around an opaque leader. **Pause** freezes training. The speed selector accelerates simulation, while lap times stay in simulation seconds. **Rays** toggles sensor drawing. Sensing always remains active.

**View network** opens layers and activations. Hover over nodes for details. Close it or press Escape to return.

**Compare networks** starts all four architectures on the same track with 64 cars, seed 42, and 100 generations each. The network selector changes inspection. **Single network** starts fresh training of the selected architecture with ghost cars. **Reset**, track changes, and changing networks outside comparison create a fresh paused run.

## How it learns

Each car has its own neural network. Weights and biases start at zero, so generation 0 stays still. Mutations create varied driving attempts from generation 1 onward.

Ordered checkpoints measure progress and fitness. A completed lap beats every incomplete attempt, then shorter times score higher. Collisions and stalling end attempts.

The strongest networks survive each generation. Other cars inherit mixed copies with random mutations. This evolutionary selection develops driving behavior and faster laps. The fixed seed makes fresh runs repeatable.

Five wall distances enter the network. Rays point 60 and 30 degrees left, straight ahead, and 30 and 60 degrees right. Inputs range from 0 to 1. Smaller values mean a nearby boundary. A value of 1 means clear space to maximum sensor range.

Outputs request steering, target speed, and braking. Physics applies acceleration or braking toward that speed. Explicit braking takes priority. Faster cars turn wider, so braking helps with tight corners.

| Network | Layers |
| --- | --- |
| Compact | 5 → 12 → 3 |
| Wide | 5 → 24 → 3 |
| Deep | 5 → 16 → 12 → 3 |
| Split steering | 5 → 20 → 12 → 4 |

Split steering uses separate left and right steering outputs.

Blue shows each generation's best fitness. Higher is better. Green shows its fastest lap. Lower is better, and gaps mean nobody finished. **Best** retains the lap record. Comparison curves show each network's best lap so far.

## Upload your track

Draw one wide, solid closed road on a contrasting background, with empty space around it. Avoid intersections, branches, markings, text, and textures. For a 960 by 640 image, start with a road at least 40 pixels wide.

Click **Upload track** or drop a static PNG or JPEG onto the track pane. The limit is 8 MiB and 8 million pixels, with both sides at least 64 pixels. Images fit within 960 by 640. Road detection, start position, and direction are automatic.

A successful upload starts a fresh paused run. Click **Train**. If detection fails, simplify the image and widen the road. The circuit selector restores built-in tracks.

## Save a model

**Save brain** exports the inspected network's best policy as JSON after a completed generation. Reloading models uses Python's `racer.models.load_model` and `Trainer.load_model`. Restarting the server clears training from memory.

Built-in circuits are widened F1-inspired approximations. Lap times are simulator comparisons, with elevation and banking omitted. Attribution is in [NOTICE](NOTICE).

## Screenshots

![Single-network training with ghost cars and wall rays](docs/images/track.jpg)

Four leaders in comparison mode. Blue is Compact, orange is Wide, green is Deep, and pink is Split steering.

![Four neural networks compared on the same track](docs/images/comparison.jpg)

Blue connections are positive and pink are negative. Stronger connections appear darker and thicker.

![Live neural network layers, activations, and driving outputs](docs/images/network.jpg)
