"""Load track images and validate a user-defined circuit on the road surface."""

from __future__ import annotations

from dataclasses import dataclass, fields
from collections import deque
from functools import lru_cache
from io import BytesIO
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError


MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 8_000_000
MAX_WIDTH = 960
MAX_HEIGHT = 640
MIN_SIDE = 64
MAX_ROUTE_POINTS = 256
MIN_ROUTE_LENGTH = 80.0
AUTO_TRACK_ERROR = "Use a top-down image with one solid closed road, a contrasting background, and no branches or intersections."


def _immutable(array, dtype):
    array = np.asarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=dtype).reshape(array.shape)


@dataclass(frozen=True)
class VehicleProfile:
    car_length: float
    car_width: float
    wheelbase: float
    max_speed: float
    ray_range: float
    max_steer_angle: float
    max_lateral_acceleration: float
    acceleration: float
    braking: float
    drag: float
    gate_spacing: float
    gate_half_width: float
    progress_threshold: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError("Vehicle profile values must be finite positive numbers.")
            object.__setattr__(self, field.name, float(value))
        if not self.car_width <= self.car_length or not self.wheelbase <= self.car_length:
            raise ValueError("The vehicle width and wheelbase must fit its length.")
        if self.max_steer_angle >= np.pi / 2 or self.ray_range < self.car_length:
            raise ValueError("The vehicle steering and sensor ranges must be valid.")
        if self.gate_half_width <= self.car_width / 2 or self.drag >= self.acceleration:
            raise ValueError("The vehicle needs road clearance and positive acceleration.")

    @classmethod
    def scaled(cls, pixels_per_metre, road_width_metres=12):
        if type(pixels_per_metre) not in (int, float) or not np.isfinite(pixels_per_metre) or pixels_per_metre <= 0:
            raise ValueError("The track scale must be a finite positive number.")
        if type(road_width_metres) not in (int, float) or not np.isfinite(road_width_metres) or road_width_metres <= 2:
            raise ValueError("The road must be wider than the vehicle.")
        scale = float(pixels_per_metre)
        #common time compression fits full-size circuits into the simulation lap limit
        return cls(car_length=5.5 * scale, car_width=2 * scale, wheelbase=3.6 * scale,
                   max_speed=9 * scale, ray_range=90 * scale, max_steer_angle=0.65,
                   max_lateral_acceleration=0.54 * scale, acceleration=0.12 * scale,
                   braking=0.45 * scale, drag=0.008 * scale, gate_spacing=8 * scale,
                   gate_half_width=road_width_metres * scale / 2, progress_threshold=0.5 * scale)


@dataclass(frozen=True)
class Track:
    mask: np.ndarray
    image: np.ndarray
    route: np.ndarray
    start: np.ndarray
    heading: float
    name: str
    vehicle: VehicleProfile | None = None

    def __post_init__(self):
        image, mask = np.asarray(self.image), np.asarray(self.mask)
        route, start = np.asarray(self.route, dtype=float), np.asarray(self.start, dtype=float)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2]:
            raise ValueError("Track image and road mask dimensions must match.")
        if route.ndim != 2 or route.shape[1] != 2 or len(route) < 4 or start.shape != (2,):
            raise ValueError("Track route and start must contain two-dimensional coordinates.")
        if not np.isfinite(route).all() or not np.isfinite(start).all() or not np.isfinite(self.heading):
            raise ValueError("Track geometry must be finite.")
        if self.vehicle is not None and not isinstance(self.vehicle, VehicleProfile):
            raise ValueError("The track vehicle profile must be a VehicleProfile.")
        for name, value, dtype in (("image", image, np.uint8), ("mask", mask, bool),
                                   ("route", route, np.float64), ("start", start, np.float64)):
            object.__setattr__(self, name, _immutable(value, dtype))
        object.__setattr__(self, "heading", float(self.heading))

    @property
    def width(self):
        return int(self.mask.shape[1])

    @property
    def height(self):
        return int(self.mask.shape[0])


def load_image(path_or_bytes) -> np.ndarray:
    """Decode a static png or jpeg and fit it into the working image bounds."""
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise ValueError("The track image could not be read.") from exc
    elif isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        raw = bytes(path_or_bytes)
    else:
        raise ValueError("Choose a PNG or JPEG image file.")
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError("Track images must be nonempty and no larger than 8 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw), formats=("PNG", "JPEG")) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Track images must not exceed 8 million pixels.")
                if min(source.size) < MIN_SIDE:
                    raise ValueError("Track images must be at least 64 pixels on each side.")
                if getattr(source, "is_animated", False):
                    raise ValueError("Choose a static PNG or JPEG image.")
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    image = Image.new("RGB", rgba.size, (0, 0, 0))
                    image.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    image = oriented.convert("RGB")
                image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
                if min(image.size) < MIN_SIDE:
                    raise ValueError("The image is too narrow after resizing. Use a less extreme aspect ratio.")
                return np.array(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("The file is not a readable static PNG or JPEG image.") from exc


def _point(value, width, height):
    try:
        point = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Coordinates must contain two finite pixel values.") from exc
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Coordinates must contain two finite pixel values.")
    if not 0 <= point[0] < width or not 0 <= point[1] < height:
        raise ValueError("A selected point lies outside the track image.")
    return point


def _cross(first, second):
    return first[0] * second[1] - first[1] * second[0]


def _intersects(first, second, third, fourth):
    delta, other = second - first, fourth - third
    determinant = _cross(delta, other)
    offset = third - first
    if abs(determinant) > 1e-9:
        t = _cross(offset, other) / determinant
        u = _cross(offset, delta) / determinant
        return -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9
    if abs(_cross(offset, delta)) > 1e-9:
        return False
    axis = int(np.argmax(np.abs(delta)))
    return max(min(first[axis], second[axis]), min(third[axis], fourth[axis])) <= min(
        max(first[axis], second[axis]), max(third[axis], fourth[axis])
    ) + 1e-9


def _route_on_mask(mask, route):
    height, width = mask.shape
    try:
        route = np.asarray(route, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("The route must be a list of finite pixel coordinate pairs.") from exc
    if route.ndim != 2 or route.shape[1] != 2 or not np.isfinite(route).all():
        raise ValueError("The route must be a list of finite pixel coordinate pairs.")
    #the closing segment is implicit, so keep the start only once
    if len(route) > 1 and np.linalg.norm(route[-1] - route[0]) <= 0.5:
        route = route[:-1]
    if not 4 <= len(route) <= MAX_ROUTE_POINTS:
        raise ValueError("Choose between 4 and 256 ordered route points around one complete lap.")
    for point in route:
        _point(point, width, height)
        if not mask[int(point[1]), int(point[0])]:
            raise ValueError("Every route point must lie on the selected road surface.")
    segments = np.roll(route, -1, axis=0) - route
    lengths = np.linalg.norm(segments, axis=1)
    if (lengths < 2).any():
        raise ValueError("Route points must be at least two pixels apart.")
    if lengths.sum() < MIN_ROUTE_LENGTH:
        raise ValueError("The complete route must be at least 80 pixels long.")
    area = abs(np.sum(route[:, 0] * np.roll(route[:, 1], -1) - route[:, 1] * np.roll(route[:, 0], -1))) / 2
    if area < 64:
        raise ValueError("The route must form a loop enclosing an area, rather than a line.")
    for index, (start, delta, length) in enumerate(zip(route, segments, lengths)):
        samples = start + np.linspace(0, 1, int(np.ceil(length / 2)) + 1)[:, None] * delta
        if not mask[samples[:, 1].astype(int), samples[:, 0].astype(int)].all():
            raise ValueError("A route segment leaves the road. Add points through each bend, including the final bend back to the start.")
        for following in range(index + 2, len(route)):
            if index == 0 and following == len(route) - 1:
                continue
            if _intersects(start, start + delta, route[following], route[(following + 1) % len(route)]):
                raise ValueError("The route must follow a single loop without crossing or retracing itself.")
    return route.copy()


def build_track(image, road_point, route, name="Custom", tolerance=48) -> Track:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Track images must contain RGB pixels.")
    height, width = image.shape[:2]
    if not MIN_SIDE <= width <= MAX_WIDTH or not MIN_SIDE <= height <= MAX_HEIGHT:
        raise ValueError("Load the image first so its dimensions fit within 960 by 640 pixels.")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
        raise ValueError("Track names must contain between 1 and 120 characters.")
    if type(tolerance) not in (float, int) or not np.isfinite(tolerance) or not 1 <= tolerance <= 128:
        raise ValueError("Road color tolerance must be a number from 1 to 128.")
    seed = _point(road_point, width, height)
    pixel = image[int(seed[1]), int(seed[0])].astype(np.int32)
    delta = image.astype(np.int32) - pixel
    candidate = np.sum(delta * delta, axis=2) <= tolerance * tolerance
    component = Image.fromarray(candidate.astype(np.uint8) * 255).copy()
    #flood only the connected surface of the chosen road color
    ImageDraw.floodfill(component, (int(seed[0]), int(seed[1])), 128)
    mask = np.asarray(component) == 128
    if mask.sum() < max(256, int(width * height * 0.001)):
        raise ValueError("The selected road surface is too small. Click near the middle of the road.")
    if mask.mean() > 0.95:
        raise ValueError("The selected color covers almost the entire image. Choose a distinct road surface.")
    route = _route_on_mask(mask, route)
    heading = np.arctan2(*(route[1] - route[0])[::-1])
    return Track(mask, image, route, route[0], heading, name.strip())


def _components(mask, minimum=1):
    work = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).copy()
    found = []
    for label in range(1, 129):
        remaining = np.flatnonzero(np.asarray(work).ravel() == 255)
        if not len(remaining):
            return found
        y, x = divmod(int(remaining[0]), mask.shape[1])
        ImageDraw.floodfill(work, (x, y), label)
        component = np.asarray(work) == label
        if component.sum() >= minimum:
            found.append(component)
    raise ValueError("The image has too many separate regions. " + AUTO_TRACK_ERROR)


def _thin(mask):
    #remove border pixels in paired passes while preserving circuit connectivity
    image = np.pad(mask, 1)
    for _ in range(max(mask.shape)):
        changed = False
        for phase in (0, 1):
            neighbors = (image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:], image[2:, 2:],
                         image[2:, 1:-1], image[2:, :-2], image[1:-1, :-2], image[:-2, :-2])
            count = np.add.reduce(neighbors, dtype=np.uint8)
            transitions = np.add.reduce([~neighbors[index] & neighbors[(index + 1) % 8] for index in range(8)], dtype=np.uint8)
            north, _, east, _, south, _, west, _ = neighbors
            if phase == 0:
                corners = ~(north & east & south) & ~(east & south & west)
            else:
                corners = ~(north & east & west) & ~(north & south & west)
            remove = image[1:-1, 1:-1] & (count >= 2) & (count <= 6) & (transitions == 1) & corners
            if remove.any():
                image[1:-1, 1:-1] &= ~remove
                changed = True
        if not changed:
            return image[1:-1, 1:-1].copy()
    raise ValueError("The road centerline could not be resolved. " + AUTO_TRACK_ERROR)


def _cycle_route(mask):
    skeleton = _thin(mask)
    height, width = mask.shape
    nodes = set(map(int, np.flatnonzero(skeleton.ravel())))
    graph = {}
    for node in nodes:
        y, x = divmod(node, width)
        adjacent = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dx or dy) or not 0 <= y + dy < height or not 0 <= x + dx < width:
                    continue
                other = (y + dy) * width + x + dx
                if other not in nodes:
                    continue
                #avoid diagonal shortcuts across an existing right-angle connection
                if dx and dy and (skeleton[y, x + dx] or skeleton[y + dy, x]):
                    continue
                adjacent.add(other)
        graph[node] = adjacent
    original_count = len(graph)
    leaves = deque(node for node, adjacent in graph.items() if len(adjacent) < 2)
    while leaves:
        node = leaves.popleft()
        for neighbor in graph.pop(node, ()):
            graph[neighbor].discard(node)
            if len(graph[neighbor]) < 2:
                leaves.append(neighbor)
    if not graph or any(len(adjacent) != 2 for adjacent in graph.values()):
        raise ValueError("The road does not have one unambiguous centerline. " + AUTO_TRACK_ERROR)
    if original_count - len(graph) > max(8, original_count * 0.03):
        raise ValueError("The road contains a branch or dead end. " + AUTO_TRACK_ERROR)
    points = np.array([[node % width, node // width] for node in graph], dtype=float)
    radius = max(2, int(np.ceil(min(width, height) * 0.021)))
    clear = np.ones(len(points), dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            x, y = points[:, 0].astype(int) + dx, points[:, 1].astype(int) + dy
            valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            clear &= valid & mask[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)]
    if not clear.any():
        raise ValueError("The road is too narrow for a safe starting position. " + AUTO_TRACK_ERROR)
    safe = points[clear]
    center_x = (points[:, 0].min() + points[:, 0].max()) / 2
    start_point = min(safe, key=lambda point: (point[1], abs(point[0] - center_x), point[0]))
    start = int(start_point[1]) * width + int(start_point[0])
    visited, ordered, previous, current = set(), [], None, start
    while current not in visited:
        visited.add(current)
        ordered.append([current % width, current // width])
        candidates = graph[current] - ({previous} if previous is not None else set())
        following = max(candidates, key=lambda node: (node % width, node // width))
        previous, current = current, following
    if current != start or len(visited) != len(graph):
        raise ValueError("The road contains multiple loops. " + AUTO_TRACK_ERROR)
    ordered = np.asarray(ordered, dtype=float)
    area = np.sum(ordered[:, 0] * np.roll(ordered[:, 1], -1) - ordered[:, 1] * np.roll(ordered[:, 0], -1))
    if area < 0:
        ordered = np.concatenate((ordered[:1], ordered[:0:-1]))
    closed = np.concatenate((ordered, ordered[:1]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    count = min(MAX_ROUTE_POINTS, max(24, int(np.ceil(cumulative[-1] / 20))))
    distances = np.arange(count) * cumulative[-1] / count
    route = np.column_stack([np.interp(distances, cumulative, closed[:, axis]) for axis in (0, 1)])
    return _route_on_mask(mask, route)


def track_from_image(image, name="Custom") -> Track:
    """Infer a single closed centerline from a high-contrast annular road."""
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Load a PNG or JPEG track image first.")
    height, width = image.shape[:2]
    if not MIN_SIDE <= width <= MAX_WIDTH or not MIN_SIDE <= height <= MAX_HEIGHT:
        raise ValueError("Load the image first so it fits within 960 by 640 pixels.")
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 120:
        raise ValueError("Track names must contain between 1 and 120 characters.")
    gray = np.rint(image.astype(float) @ np.array([0.2126, 0.7152, 0.0722])).astype(np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(float)
    probability = histogram / histogram.sum()
    weight = np.cumsum(probability)
    mean = np.cumsum(probability * np.arange(256))
    variance = np.divide((mean[-1] * weight - mean) ** 2, weight * (1 - weight),
                         out=np.zeros(256), where=(weight > 0) & (weight < 1))
    split = int(np.argmax(variance))
    low, high = gray[gray <= split], gray[gray > split]
    if not len(low) or not len(high) or high.mean() - low.mean() < 32:
        raise ValueError(AUTO_TRACK_ERROR)
    #use class means so compressed noise near a flat color cannot fragment the mask
    threshold = (float(low.mean()) + float(high.mean())) / 2
    candidates, last_error = [], None
    for binary in (gray > threshold, gray <= threshold):
        for component in _components(binary, minimum=max(256, int(width * height * 0.001))):
            if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
                continue
            inverse = Image.fromarray((~component).astype(np.uint8) * 255).copy()
            ImageDraw.floodfill(inverse, (0, 0), 128)
            holes = _components(np.asarray(inverse) == 255, minimum=max(64, int(width * height * 0.001)))
            if len(holes) != 1:
                continue
            try:
                candidates.append((component, _cycle_route(component)))
            except ValueError as exc:
                last_error = exc
    if len(candidates) != 1:
        message = str(last_error) if not candidates and last_error else AUTO_TRACK_ERROR
        if AUTO_TRACK_ERROR not in message:
            message += " " + AUTO_TRACK_ERROR
        raise ValueError(message)
    mask, route = candidates[0]
    heading = np.arctan2(*(route[1] - route[0])[::-1])
    return Track(mask, image, route, route[0], heading, name.strip())


@lru_cache(maxsize=1)
def sample_track() -> Track:
    image = Image.new("RGB", (900, 560), (22, 27, 31))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((55, 55, 845, 505), fill=(224, 228, 232))
    drawing.ellipse((155, 155, 745, 405), fill=(22, 27, 31))
    angles = np.arange(24) * (2 * np.pi / 24) - np.pi / 2
    route = np.column_stack((450 + 345 * np.cos(angles), 280 + 175 * np.sin(angles)))
    return build_track(np.array(image), route[0], route, name="Oval circuit")


def save_sample(path):
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Save the sample track with a .png extension.")
    content = BytesIO()
    Image.fromarray(sample_track().image).save(content, format="PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    #exclusive creation preserves an existing user file
    with path.open("xb") as stream:
        stream.write(content.getvalue())
    return path
