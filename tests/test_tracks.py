from dataclasses import FrozenInstanceError
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
import pytest

from racer.tracks import (
    MAX_FILE_BYTES, MAX_HEIGHT, MAX_WIDTH, Track, build_track, load_image,
    sample_track, save_sample, track_from_image,
)


def ring(width=320, height=240, *, inverse=False, color=(224, 228, 232), road_width=35):
    image = Image.new("RGB", (width, height), (22, 27, 31))
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 20, width - 20, height - 20), fill=color)
    draw.ellipse((20 + road_width, 20 + road_width, width - 20 - road_width, height - 20 - road_width), fill=(22, 27, 31))
    array = np.array(image)
    return 255 - array if inverse else array


def route(width=320, height=240):
    angle = np.arange(24) * (2 * np.pi / 24) - np.pi / 2
    return np.column_stack((width / 2 + (width / 2 - 37.5) * np.cos(angle),
                            height / 2 + (height / 2 - 37.5) * np.sin(angle)))


def encode(image, format="PNG", **options):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format=format, **options)
    return stream.getvalue()


def assert_circuit(track):
    assert track.image.shape == (track.height, track.width, 3)
    assert track.image.dtype == np.uint8 and track.mask.dtype == bool
    assert len(track.route) >= 4
    assert np.isfinite(track.route).all()
    np.testing.assert_array_equal(track.start, track.route[0])
    assert not np.array_equal(track.route[0], track.route[-1])
    first = track.route[1] - track.route[0]
    assert track.heading == pytest.approx(np.arctan2(first[1], first[0]))
    area = np.sum(track.route[:, 0] * np.roll(track.route[:, 1], -1) - track.route[:, 1] * np.roll(track.route[:, 0], -1))
    assert area > 0
    for start, end in zip(track.route, np.roll(track.route, -1, axis=0)):
        steps = int(np.ceil(np.linalg.norm(end - start))) + 1
        points = np.linspace(start, end, steps)
        assert track.mask[points[:, 1].astype(int), points[:, 0].astype(int)].all()


def test_built_sample_is_a_complete_clockwise_road_circuit():
    track = sample_track()
    assert (track.width, track.height) == (900, 560)
    assert len(track.route) == 24
    assert 0.1 < track.mask.mean() < 0.6
    assert not track.mask[0, 0] and not track.mask[280, 450]
    assert track.mask[105, 450]
    assert_circuit(track)


def test_sample_png_can_be_loaded_and_inferred_without_route_annotations(tmp_path):
    path = save_sample(tmp_path / "sample.png")
    image = load_image(path)
    inferred = track_from_image(image, name="Uploaded oval")
    assert inferred.name == "Uploaded oval"
    assert_circuit(inferred)
    np.testing.assert_array_equal(inferred.mask, sample_track().mask)
    assert len(inferred.route) > 24
    with Image.open(path) as native:
        assert native.format == "PNG"
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_sample(path)
    assert path.read_bytes() == before


def test_sample_destination_requires_png_extension(tmp_path):
    with pytest.raises(ValueError):
        save_sample(tmp_path / "sample.jpg")


@pytest.mark.parametrize("inverse,color", [(False, (224, 228, 232)), (True, (224, 228, 232)), (False, (160, 105, 75))])
def test_automatic_road_detection_supports_polarity_and_colored_surfaces(inverse, color):
    track = track_from_image(ring(inverse=inverse, color=color))
    assert_circuit(track)
    assert track.mask[37, 160]
    assert not track.mask[120, 160]


def test_automatic_detection_tolerates_jpeg_compression():
    image = load_image(encode(ring(), format="JPEG", quality=65))
    track = track_from_image(image)
    assert_circuit(track)
    assert track.mask[37, 160]


def test_non_elliptical_rounded_rectangle_is_inferred_from_pixels():
    image = Image.new("RGB", (400, 300), (15, 20, 25))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((20, 20, 380, 280), radius=65, fill=(230, 230, 230))
    draw.rounded_rectangle((60, 60, 340, 240), radius=25, fill=(15, 20, 25))
    track = track_from_image(np.array(image))
    assert_circuit(track)
    assert track.mask[150, 40] and not track.mask[150, 200]


@pytest.mark.parametrize("kind", ["solid", "disk", "open", "two_loops", "branch", "narrow"])
def test_automatic_inference_rejects_noncircuits_ambiguity_and_unsafe_roads(kind):
    image = Image.fromarray(ring())
    draw = ImageDraw.Draw(image)
    if kind == "solid":
        image = Image.new("RGB", image.size, (220, 220, 220))
    elif kind == "disk":
        draw.ellipse((55, 55, 265, 185), fill=(224, 228, 232))
    elif kind == "open":
        draw.rectangle((150, 0, 175, 80), fill=(22, 27, 31))
    elif kind == "two_loops":
        image = Image.new("RGB", (640, 240), (22, 27, 31))
        image.paste(Image.fromarray(ring()), (0, 0))
        image.paste(Image.fromarray(ring()), (320, 0))
    elif kind == "branch":
        draw.rectangle((285, 112, 312, 128), fill=(224, 228, 232))
    elif kind == "narrow":
        image = Image.fromarray(ring(road_width=4))
    with pytest.raises(ValueError, match="top-down image"):
        track_from_image(np.array(image))


def test_track_arrays_are_detached_and_immutable():
    image = ring()
    points = route()
    track = build_track(image, points[0], points)
    image[:] = 0
    points[:] = 0
    assert track.image.any() and track.route.any()
    with pytest.raises(FrozenInstanceError):
        track.name = "Changed"
    for array in (track.image, track.mask, track.route, track.start):
        with pytest.raises(ValueError):
            array.flat[0] = 0
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_manual_route_has_implicit_closing_segment_without_duplicate_start():
    points = route()
    track = build_track(ring(), points[0], np.vstack((points, points[0])), name="  Custom loop  ")
    assert track.name == "Custom loop"
    assert len(track.route) == 24
    assert_circuit(track)


def test_color_floodfill_excludes_disconnected_same_color_decorations():
    image = ring()
    image[2:10, 2:10] = image[37, 160]
    track = build_track(image, [160, 37], route())
    assert not track.mask[5, 5]
    assert track.mask[37, 160]


@pytest.mark.parametrize("points", [
    [], [[1, 2], [3, 4], [5, 6]], [[160, 37], [np.nan, 50], [280, 120], [160, 202]],
    [[160, 37], [200, 120], [160, 202], [37, 120]],
    [[160, 37], [37, 120], [280, 120], [160, 202]],
    [[160, 37], [161, 37], [280, 120], [160, 202]],
    [[-1, 37], [37, 120], [160, 202], [280, 120]],
])
def test_manual_routes_must_be_finite_on_road_and_close_without_shortcuts(points):
    with pytest.raises(ValueError):
        build_track(ring(), [160, 37], points)


def test_route_validation_checks_final_bend_back_to_start():
    points = route()[:16]
    with pytest.raises(ValueError, match="segment leaves the road"):
        build_track(ring(), points[0], points)


@pytest.mark.parametrize("point", [[-1, 0], [320, 20], [160, float("inf")], [1], "x"])
def test_invalid_road_seeds_are_rejected(point):
    with pytest.raises(ValueError):
        build_track(ring(), point, route())


@pytest.mark.parametrize("tolerance", [0, 129, True, float("nan"), "48"])
def test_color_tolerance_is_bounded(tolerance):
    with pytest.raises(ValueError):
        build_track(ring(), [160, 37], route(), tolerance=tolerance)


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_image_loader_preserves_aspect_ratio_and_normalizes_working_size(format):
    original = np.full((1200, 1800, 3), 128, dtype=np.uint8)
    loaded = load_image(encode(original, format=format))
    assert loaded.dtype == np.uint8 and loaded.shape[2] == 3
    assert loaded.shape[:2] == (640, 960)
    assert loaded.shape[1] <= MAX_WIDTH and loaded.shape[0] <= MAX_HEIGHT


def test_loader_composites_transparency_and_applies_exif_orientation():
    transparent = np.full((80, 120, 4), 255, dtype=np.uint8)
    transparent[:, :, 3] = 0
    assert not load_image(encode(transparent)).any()
    image = Image.new("RGB", (120, 80), (100, 150, 200))
    exif = Image.Exif()
    exif[274] = 6
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=exif)
    assert load_image(stream.getvalue()).shape == (120, 80, 3)


@pytest.mark.parametrize("content", [b"", b"not an image", b"x" * (MAX_FILE_BYTES + 1)])
def test_invalid_or_oversized_encoded_images_are_rejected(content):
    with pytest.raises(ValueError):
        load_image(content)


def test_image_loader_rejects_tiny_pixels_extreme_aspect_and_large_pixel_count():
    for shape in ((63, 100, 3), (100, 8000, 3), (2001, 4000, 3)):
        with pytest.raises(ValueError):
            load_image(encode(np.zeros(shape, dtype=np.uint8)))


def test_unsupported_image_formats_and_missing_paths_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_image(encode(ring(), format="GIF"))
    with pytest.raises(ValueError):
        load_image(tmp_path / "missing.png")


def test_track_constructor_rejects_inconsistent_geometry():
    base = sample_track()
    with pytest.raises(ValueError):
        Track(base.mask[:10], base.image, base.route, base.start, base.heading, "Invalid")
    with pytest.raises(ValueError):
        Track(base.mask, base.image, np.array([[0, 0]]), base.start, base.heading, "Invalid")
