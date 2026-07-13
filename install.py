"""ComfyUI-Manager post-install hook.

ComfyUI-Manager runs this script from the node pack root after installing
``requirements.txt``. It provisions the optimized native kernel package for
the current runtime (downloading the matching prebuilt variant wheel from the
OrbitQuant GitHub release). Provisioning is best effort: when no variant fits
this runtime, OrbitQuant falls back to its Triton or dequantized paths, so
this script never fails the node installation.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys

_PREFIX = "[ComfyUI-OrbitQuant]"
_TIMEOUT_SECONDS = 600


def main() -> int:
    try:
        import orbitquant  # noqa: F401
    except ImportError:
        print(
            f"{_PREFIX} the orbitquant package is not importable; skipping native "
            "kernel provisioning. Install the node pack requirements first.",
            file=sys.stderr,
        )
        return 0

    print(f"{_PREFIX} provisioning the native OrbitQuant kernel package...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "orbitquant.cli.main", "kernels-install"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"{_PREFIX} native kernel provisioning did not run: {exc}", file=sys.stderr)
        return 0

    report = None
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        report = json.loads(result.stdout)

    if result.returncode == 0 and report is not None:
        print(
            f"{_PREFIX} native kernel package ready "
            f"(source={report.get('source')}, variant={report.get('variant')})."
        )
    else:
        detail = (report or {}).get("detail") or result.stderr.strip() or result.stdout.strip()
        print(
            f"{_PREFIX} no prebuilt native kernel variant matched this runtime; "
            "packed runtime modes will use the Triton or dequantized fallbacks. "
            f"Detail: {detail}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
