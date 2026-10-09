"""Published model recipes. Importing the catalog does not access the network or GPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Recipe:
    repo_id: str
    adapter: str
    media_type: str = "image"
    width: int = 1024
    height: int = 1024
    steps: int = 4
    frames: int = 1
    fps: int = 24
    guidance: float = 1.0
    reference_images: bool = False

    def to_dict(self):
        return asdict(self)


def _catalog() -> dict[str, Recipe]:
    recipes = [
        Recipe(
            "WaveCut/Kandinsky-6.0-Lite-distill-5s-OrbitQuant-W4A4",
            "kandinsky6",
            "video",
            864,
            480,
            10,
            121,
        )
    ]
    for family, adapter, bits, options in (
        (
            "FLUX.2-klein-4B",
            "flux2_component",
            ("W4A4", "W3A3", "W2A4", "W2A3"),
            {"reference_images": True},
        ),
        ("FLUX.1-schnell", "flux_component", ("W4A4", "W3A3", "W2A4", "W2A3"), {"guidance": 0.0}),
        (
            "Z-Image-Turbo",
            "z_image_component",
            ("W4A4", "W3A3", "W2A4", "W2A3"),
            {"steps": 10, "guidance": 0.0},
        ),
        (
            "Wan2.1-T2V-1.3B-Diffusers",
            "wan_component",
            ("W4A6", "W4A4"),
            {
                "media_type": "video",
                "width": 832,
                "height": 480,
                "steps": 50,
                "frames": 81,
                "fps": 16,
                "guidance": 5.0,
            },
        ),
        ("FLUX.2-klein-9B", "flux2", ("W4A4",), {"reference_images": True}),
        (
            "Ideogram-v4-Instant",
            "ideogram4",
            ("W4A4", "W3A3", "W2A4", "W2A3", "W4A6"),
            {"steps": 8},
        ),
        (
            "MiniMax-H3",
            "minimax_h3",
            ("W4A4",),
            {
                "media_type": "video",
                "width": 608,
                "height": 480,
                "steps": 24,
                "frames": 124,
                "reference_images": True,
            },
        ),
        ("Krea-2-Turbo", "krea2", ("W4A4",), {"steps": 8}),
        ("YuE2-3B", "yue2", ("W4A4",), {"media_type": "audio", "steps": 8}),
        ("Turbo-Image-2.1", "qwen_image21", ("W4A4",), {"steps": 6, "reference_images": True}),
        ("Boogu-Image-0.1-Turbo", "boogu", ("W4A8",), {"reference_images": True}),
    ):
        recipes.extend(
            Recipe(f"WaveCut/{family}-OrbitQuant-{bit}", adapter, **options) for bit in bits
        )
    return {recipe.repo_id: recipe for recipe in recipes}


CATALOG = _catalog()

# These revisions identify source components, not the selectable quantized release.
IDEOGRAM_COMPONENTS = (
    "ideogram-ai/ideogram-4-nf4-diffusers",
    "1874bc70267ba2c823a7239e1d70dd308c8d64dc",
)
BOOGU_SOURCE = ("Boogu/Boogu-Image-0.1-Turbo", "47cd26d3c211b030c64ffe60d63b2e17a7c519c2")
TURBO_SIGMAS = {
    4: [1.0, 0.75, 0.5, 0.25],
    5: [1.0, 0.875, 0.75, 0.5, 0.25],
    6: [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
    7: [1.0, 0.9583, 0.9167, 0.875, 0.75, 0.5, 0.25],
    8: [1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125],
}
