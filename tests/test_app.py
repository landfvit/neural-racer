import io
import math
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from app import MAX_FILE_BYTES, create_app
from racer.service import Workbench
from racer.tracks import sample_track


@pytest.fixture
def client():
    with TestClient(create_app(live=False)) as current:
        yield current


def image_bytes(pixels):
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def test_viewer_starts_with_real_cars_and_no_documentation_page(client):
    assert client.get("/").status_code == 200
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404
    state = client.get("/api/state").json()
    assert len(state["cars"]) == 64
    assert len(state["brain"]["inputs"]) == 5
    assert len(state["brain"]["hidden"]) == 12
    assert len(state["brain"]["outputs"]) == 3
    assert state["network"]["outputs"] == ["steering", "target_speed", "brake"]
    assert "speed" not in state["network"]["inputs"]
    assert state["controls"] == {"steering": 0, "target_speed": 0, "throttle": 0, "brake": 0}
    assert all(car["throttle"] == 0 and car["brake"] == 0 for car in state["cars"])
    assert not state["running"]
    assert state["metrics"]["best_time"] is None
    assert state["metrics"]["best_ever_time"] is None
    assert all(car["elapsed_time"] == 0 and car["lap_time"] is None and not car["finished"] for car in state["cars"])
    response = client.get(state["track"]["image_url"])
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (state["track"]["width"], state["track"]["height"])


def test_training_advances_python_physics_and_pause_stops_it(client):
    bench = client.app.state.workbench
    first = client.post("/api/train", json={"running": True}).json()
    bench.advance()
    moved = client.get("/api/state").json()
    assert moved["step_count"] == first["step_count"] + first["speed"]
    assert moved["sequence"] > first["sequence"]
    client.post("/api/train", json={"running": False})
    frozen = bench.snapshot()
    bench.advance()
    assert bench.snapshot() == frozen


@pytest.mark.parametrize("speed", [1, 4, 16])
def test_viewer_speed_advances_simulation_time_without_changing_clock_units(client, speed):
    previous = client.get("/api/state").json()
    chosen = client.post("/api/speed", json={"speed": speed}).json()
    assert chosen["speed"] == speed
    assert chosen["revision"] == previous["revision"]
    client.post("/api/train", json={"running": True})
    client.app.state.workbench.advance()
    advanced = client.get("/api/state").json()
    assert advanced["step_count"] == speed
    assert all(0 < car["elapsed_time"] <= speed / 30 for car in advanced["cars"])
    assert all(car["elapsed_time"] == pytest.approx(speed / 30) for car in advanced["cars"] if car["alive"])
    assert all(car["elapsed_time"] * 30 == pytest.approx(round(car["elapsed_time"] * 30))
               for car in advanced["cars"])


@pytest.mark.parametrize("payload", [{"speed": 2}, {"speed": True}, {"speed": "16"}, {"speed": 1, "extra": 1}])
def test_invalid_speed_does_not_change_working_run(client, payload):
    before = client.get("/api/state").json()
    assert client.post("/api/speed", json=payload).status_code == 422
    assert client.get("/api/state").json() == before


def test_reset_preserves_track_but_invalidates_old_frames_and_images(client):
    before = client.get("/api/state").json()
    client.post("/api/train", json={"running": True})
    client.app.state.workbench.advance()
    reset = client.post("/api/reset").json()
    assert reset["instance"] == before["instance"]
    assert reset["sequence"] > before["sequence"]
    assert reset["revision"] > before["revision"]
    assert reset["track"]["name"] == before["track"]["name"]
    assert reset["track"]["circuit_id"] == before["track"]["circuit_id"]
    assert reset["step_count"] == 0 and not reset["running"]
    assert client.get(before["track"]["image_url"]).status_code == 404


def test_image_upload_builds_a_pixel_derived_circuit_and_resets_training(client):
    before = client.get("/api/state").json()
    data = image_bytes(sample_track().image)
    response = client.post("/api/track", files={"file": ("my-track.png", data, "image/png")})
    assert response.status_code == 200, response.text
    uploaded = response.json()
    assert uploaded["track"]["name"] == "my-track.png"
    assert uploaded["track"]["circuit_id"] is None
    assert uploaded["revision"] > before["revision"]
    assert uploaded["sequence"] > before["sequence"]
    assert len(uploaded["track"]["route"]) >= 4
    assert uploaded["track"]["route"] != before["track"]["route"]
    assert uploaded["step_count"] == 0 and not uploaded["running"]
    assert client.post("/api/train", json={"running": True}).status_code == 200
    client.app.state.workbench.advance()
    assert client.get("/api/state").json()["step_count"] > 0


def test_circuit_catalog_exposes_the_available_difficulties(client):
    response = client.get("/api/circuits")
    assert response.status_code == 200
    circuits = response.json()["circuits"]
    assert [item["id"] for item in circuits] == ["monza", "silverstone", "spa", "monaco", "interlagos"]
    assert len(circuits) == 5
    assert all(item["name"] and item["difficulty"] for item in circuits)


def test_architecture_catalog_exposes_distinct_trainable_networks(client):
    response = client.get("/api/architectures")
    assert response.status_code == 200
    architectures = response.json()["architectures"]
    assert [item["id"] for item in architectures] == ["compact", "wide", "deep", "split"]
    assert [item["sizes"] for item in architectures] == [[5, 12, 3], [5, 24, 3], [5, 16, 12, 3], [5, 20, 12, 4]]
    assert architectures[-1]["outputs"] == ["steering_left", "steering_right", "target_speed", "brake"]
    assert len({item["parameters"] for item in architectures}) == 4


@pytest.mark.parametrize("architecture_id", ["compact", "wide", "deep", "split"])
def test_selecting_a_network_starts_a_fresh_population_on_the_same_track(client, architecture_id):
    client.post("/api/circuit", json={"id": "spa"})
    client.post("/api/train", json={"running": True})
    client.app.state.workbench.advance()
    before = client.get("/api/state").json()
    selected = client.post("/api/architecture", json={"id": architecture_id})
    assert selected.status_code == 200, selected.text
    state = selected.json()
    assert state["network"]["id"] == architecture_id
    assert state["track"]["name"] == before["track"]["name"]
    assert state["track"]["circuit_id"] == "spa"
    assert state["track"]["route"] == before["track"]["route"]
    assert state["revision"] > before["revision"] and state["sequence"] > before["sequence"]
    assert state["generation"] == state["step_count"] == 0
    assert not state["running"] and state["history"] == []
    assert state["metrics"]["best_ever_time"] is None
    assert len(state["brain"]["outputs"]) == len(state["network"]["outputs"])
    assert [len(layer) for layer in state["brain"]["hidden_layers"]] == state["network"]["sizes"][1:-1]
    client.post("/api/train", json={"running": True})
    client.app.state.workbench.advance()
    advanced = client.get("/api/state").json()
    assert advanced["step_count"] > 0
    assert all(-1 <= car["steering"] <= 1 and 0 <= car["brake"] <= 1 and 0 <= car["target_speed"] <= 1
               for car in advanced["cars"])
    model = client.get("/api/model").json()
    assert model["architecture_id"] == architecture_id
    assert len(model["weights"]) == state["network"]["parameters"]


def test_network_choice_survives_reset_circuit_change_and_upload(client):
    client.post("/api/architecture", json={"id": "split"})
    assert client.post("/api/reset").json()["network"]["id"] == "split"
    selected = client.post("/api/circuit", json={"id": "silverstone"}).json()
    assert selected["network"]["id"] == "split"
    data = image_bytes(sample_track().image)
    uploaded = client.post("/api/track", files={"file": ("custom.png", data, "image/png")})
    assert uploaded.status_code == 200
    assert uploaded.json()["network"]["id"] == "split"
    assert uploaded.json()["track"]["circuit_id"] is None


@pytest.mark.parametrize("payload", [{"id": "unknown"}, {"id": "../../model"}, {"id": 12}, {"id": "deep", "hidden": 50}])
def test_invalid_network_selection_preserves_the_run(client, payload):
    client.post("/api/train", json={"running": True})
    before = client.get("/api/state").json()
    assert client.post("/api/architecture", json=payload).status_code == 422
    assert client.get("/api/state").json() == before


def test_comparison_starts_all_networks_with_the_same_track_and_budget(client):
    client.post("/api/circuit", json={"id": "silverstone"})
    client.post("/api/architecture", json={"id": "deep"})
    before = client.get("/api/state").json()
    assert before["comparison"] is None
    response = client.post("/api/comparison")
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["running"] and not state["complete"]
    assert state["network"]["id"] == "deep"
    assert state["track"]["route"] == before["track"]["route"]
    assert state["track"]["circuit_id"] == "silverstone"
    assert state["revision"] > before["revision"]
    comparison = state["comparison"]
    assert comparison["status"] == "running"
    assert comparison["seed"] == 42 and comparison["population"] == 64
    assert comparison["generation_limit"] == 100 and comparison["completed_generations"] == 0
    assert [entry["id"] for entry in comparison["entries"]] == ["compact", "wide", "deep", "split"]
    assert all(entry["history"] == [] and entry["best_time"] is None and entry["first_lap_generation"] is None
               for entry in comparison["entries"])
    client.app.state.workbench.advance()
    advanced = client.get("/api/state").json()
    assert advanced["step_count"] == advanced["speed"]


def test_comparison_inspection_pause_and_model_download_keep_the_shared_run(client):
    client.post("/api/comparison")
    client.app.state.workbench.advance()
    before = client.get("/api/state").json()
    selected = client.post("/api/architecture", json={"id": "split"}).json()
    assert selected["running"]
    assert selected["revision"] == before["revision"]
    assert selected["comparison"] == before["comparison"]
    assert selected["network"]["id"] == "split"
    assert selected["step_count"] == before["step_count"]
    assert client.get("/api/model").json()["architecture_id"] == "split"
    paused = client.post("/api/train", json={"running": False}).json()
    assert paused["comparison"]["status"] == "paused"
    client.app.state.workbench.advance()
    assert client.get("/api/state").json() == paused
    resumed = client.post("/api/train", json={"running": True}).json()
    assert resumed["comparison"]["status"] == "running"
    reset = client.post("/api/reset").json()
    assert reset["comparison"] is None and not reset["running"]
    assert reset["network"]["id"] == "split" and reset["step_count"] == 0


def test_track_change_discards_comparison_results_for_the_old_circuit(client):
    client.post("/api/architecture", json={"id": "wide"})
    client.post("/api/comparison")
    selected = client.post("/api/circuit", json={"id": "monaco"}).json()
    assert selected["comparison"] is None
    assert selected["track"]["circuit_id"] == "monaco" and selected["network"]["id"] == "wide"
    assert not selected["running"]


@pytest.mark.parametrize("circuit_id", ["monza", "silverstone", "monaco", "spa", "interlagos"])
def test_selecting_a_circuit_replaces_and_pauses_the_run(client, circuit_id):
    client.post("/api/train", json={"running": True})
    client.app.state.workbench.advance()
    before = client.get("/api/state").json()
    response = client.post("/api/circuit", json={"id": circuit_id})
    assert response.status_code == 200, response.text
    selected = response.json()
    assert selected["track"]["circuit_id"] == circuit_id
    assert selected["track"]["difficulty"]
    assert selected["revision"] > before["revision"]
    assert selected["sequence"] > before["sequence"]
    assert not selected["running"] and selected["generation"] == 0 and selected["step_count"] == 0
    assert selected["history"] == [] and selected["metrics"]["best_ever_time"] is None
    assert client.get(before["track"]["image_url"]).status_code == 404
    assert client.get(selected["track"]["image_url"]).status_code == 200
    client.post("/api/train", json={"running": True})
    client.app.state.workbench.advance()
    assert client.get("/api/state").json()["step_count"] > 0


@pytest.mark.parametrize("payload", [{"id": "unknown"}, {"id": "../../examples/sample-track.png"}, {"id": 3}])
def test_invalid_circuit_selection_preserves_the_working_run(client, payload):
    client.post("/api/train", json={"running": True})
    before = client.get("/api/state").json()
    assert client.post("/api/circuit", json=payload).status_code == 422
    assert client.get("/api/state").json() == before


@pytest.mark.parametrize("data", [b"not an image", image_bytes(np.full((120, 180, 3), 255, dtype=np.uint8))])
def test_bad_upload_leaves_the_previous_track_and_training_intact(client, data):
    client.post("/api/train", json={"running": True})
    before = client.get("/api/state").json()
    response = client.post("/api/track", files={"file": ("broken.png", data, "image/png")})
    assert response.status_code == 422
    assert response.json()["error"]["message"]
    assert client.get("/api/state").json() == before


def test_upload_limit_rejects_before_image_decoding(client):
    response = client.post("/api/track", files={"file": ("large.png", b"x" * (MAX_FILE_BYTES + 1), "image/png")})
    assert response.status_code == 413


def test_model_download_is_finite_json_with_the_actual_architecture(client):
    response = client.get("/api/model")
    model = response.json()
    assert "attachment" in response.headers["content-disposition"]
    assert model["architecture"] == {"inputs": 5, "hidden": 12, "outputs": 3, "activation": "tanh"}
    assert model["version"] == 2
    assert len(model["weights"]) == 111
    assert model["outputs"] == ["steering", "target_speed", "brake"]
    assert all(math.isfinite(value) for value in model["weights"])
    assert model["objective"] == "minimum_lap_time"
    assert model["simulation_hz"] == 30
    assert model["lap_limit"] == 1


def test_websocket_delivers_the_same_live_python_state(client):
    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as socket:
        state = socket.receive_json()
        assert state == client.get("/api/state").json()


def test_external_origins_and_hosts_cannot_control_training(client):
    assert client.post("/api/train", json={"running": True}, headers={"origin": "https://example.com"}).status_code == 403
    assert client.get("/api/state", headers={"host": "example.com"}).status_code == 400
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect("/ws", headers={"origin": "https://example.com"}):
            pass
    assert error.value.code == 1008


def test_training_worker_can_restart_without_resetting_the_run():
    bench = Workbench()
    bench.set_running(True)
    advanced = threading.Event()
    original_advance = bench.advance
    previous_step = 0

    def observe_progress():
        original_advance()
        with bench.lock:
            if bench.trainer.step_count > previous_step:
                advanced.set()

    bench.advance = observe_progress
    try:
        bench.start()
        assert advanced.wait(timeout=2)
        bench.stop()
        previous_step = bench.trainer.step_count
        advanced.clear()
        bench.start()
        assert advanced.wait(timeout=2)
        bench.stop()
        assert bench.trainer.step_count > previous_step
    finally:
        bench.stop()
