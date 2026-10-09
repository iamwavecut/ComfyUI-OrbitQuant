import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from comfyui_orbitquant import comfy_entrypoint  # noqa: E402

# ComfyUI checks V1 mappings before the V3 entrypoint. Expose only one interface.
try:
    from comfy_api.latest import ComfyExtension  # noqa: F401
except ImportError:
    from comfyui_orbitquant import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
else:
    __all__ = ["comfy_entrypoint"]
