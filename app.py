from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from racer.service import Workbench
from racer.circuits import circuit_catalog
from racer.networks import architecture_catalog
from racer.tracks import load_image, track_from_image


ROOT = Path(__file__).resolve().parent
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES = MAX_FILE_BYTES + 1024 * 1024


def same_origin(headers, scheme):
    if "origin" not in headers:
        return True
    try:
        host = urlsplit(f"//{headers.get('host', '')}")
        origin = urlsplit(headers["origin"])
        scheme = {"ws": "http", "wss": "https"}.get(scheme, scheme)
        default_port = 443 if scheme == "https" else 80
        return (origin.scheme == scheme and origin.hostname == host.hostname
                and (origin.port or default_port) == (host.port or default_port)
                and not origin.path and not origin.query and not origin.fragment and origin.username is None)
    except ValueError:
        return False


class LocalRequests:
    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            return await self.application(scope, receive, send)
        headers = {key.decode("latin1"): value.decode("latin1") for key, value in scope["headers"]}
        if not same_origin(headers, scope.get("scheme", "http")):
            return await JSONResponse({"error": {"message": "Use this app from its own localhost page."}}, status_code=403)(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY_BYTES:
                return await JSONResponse({"error": {"message": "Upload an image smaller than 8 MB."}}, status_code=413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay_body():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.application(scope, replay_body, send)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    running: StrictBool


class SpeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speed: StrictInt


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    generations: StrictInt


class SeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: StrictInt


class CircuitRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr


class ArchitectureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: StrictStr


def create_app(*, live=True):
    workbench = Workbench()

    @asynccontextmanager
    async def lifespan(_application):
        if live:
            workbench.start()
        try:
            yield
        finally:
            workbench.stop()

    application = FastAPI(title="Neural Racer", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    application.state.workbench = workbench
    application.add_middleware(LocalRequests)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"])

    @application.exception_handler(ValueError)
    async def invalid_input(_request, error):
        return JSONResponse({"error": {"message": str(error)}}, status_code=422)

    @application.exception_handler(HTTPException)
    async def http_error(_request, error):
        return JSONResponse({"error": {"message": str(error.detail)}}, status_code=error.status_code)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, _error):
        return JSONResponse({"error": {"message": "Check the uploaded file or selected control."}}, status_code=422)

    @application.get("/")
    def index():
        homepage = ROOT / "static" / "site" / "index.html"
        path = homepage if homepage.is_file() else ROOT / "static" / "index.html"
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @application.get("/workbench")
    def workbench_page():
        return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/api/state")
    def state():
        return workbench.snapshot()

    @application.get("/api/circuits")
    def circuits():
        return {"circuits": circuit_catalog()}

    @application.get("/api/architectures")
    def architectures():
        return {"architectures": architecture_catalog()}

    @application.get("/api/track/image")
    def image(revision: int | None = Query(None, ge=1)):
        try:
            data = workbench.image(revision)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        return Response(data, media_type="image/png", headers={"Cache-Control": "no-store"})

    @application.post("/api/track")
    async def upload_track(file: UploadFile = File(...)):
        try:
            data = await file.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, "Upload an image smaller than 8 MB.")
            name = Path((file.filename or "Custom track").replace("\\", "/")).name[:100]
            pixels = await asyncio.to_thread(load_image, data)
            track = await asyncio.to_thread(track_from_image, pixels, name=name)
            return await asyncio.to_thread(workbench.replace_track, track)
        finally:
            await file.close()

    @application.post("/api/circuit")
    def circuit(body: CircuitRequest):
        return workbench.select_circuit(body.id)

    @application.post("/api/architecture")
    def architecture(body: ArchitectureRequest):
        return workbench.select_architecture(body.id)

    @application.post("/api/comparison")
    def comparison():
        return workbench.start_comparison()

    @application.post("/api/reset")
    def reset():
        return workbench.reset()

    @application.post("/api/train")
    def train(body: RunRequest):
        return workbench.set_running(body.running)

    @application.post("/api/speed")
    def speed(body: SpeedRequest):
        if body.speed not in {1, 4, 16}:
            raise ValueError("Choose 1x, 4x, or 16x speed.")
        return workbench.set_speed(body.speed)

    @application.post("/api/generations")
    def generations(body: GenerationRequest):
        return workbench.set_generations(body.generations)

    @application.post("/api/seed")
    def seed(body: SeedRequest):
        return workbench.set_seed(body.seed)

    @application.get("/api/model")
    def model():
        return JSONResponse(workbench.model(), headers={"Content-Disposition": 'attachment; filename="neural-racer-model.json"'})

    @application.websocket("/ws")
    async def updates(socket: WebSocket):
        if not same_origin(socket.headers, socket.scope.get("scheme", "ws")):
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while True:
                await socket.send_json(await asyncio.to_thread(workbench.snapshot))
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            pass

    application.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
    return application


app = create_app()
