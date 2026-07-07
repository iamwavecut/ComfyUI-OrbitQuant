"""ComfyUI custom nodes for OrbitQuant artifacts."""

from comfyui_orbitquant.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


async def comfy_entrypoint():
    from comfyui_orbitquant.v3 import comfy_entrypoint as v3_entrypoint

    return await v3_entrypoint()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "comfy_entrypoint"]
