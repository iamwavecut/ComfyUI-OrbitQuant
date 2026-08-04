#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def request_json(base_url: str, route: str, payload: Any | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base_url.rstrip('/')}{route}", data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download_outputs(base_url: str, history: dict[str, Any], destination: Path) -> list[str]:
    downloaded = []
    for node_output in history.get("outputs", {}).values():
        for entry in node_output.get("images", []):
            query = urllib.parse.urlencode(
                {
                    "filename": entry["filename"],
                    "subfolder": entry.get("subfolder", ""),
                    "type": entry.get("type", "output"),
                }
            )
            target = destination / entry["filename"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(
                f"{base_url.rstrip('/')}/view?{query}", timeout=600
            ) as response:
                temporary = target.with_name(f".{target.name}.tmp")
                temporary.write_bytes(response.read())
                temporary.replace(target)
            downloaded.append(str(target))
    return downloaded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    prompt = json.loads(args.workflow.read_text())
    evidence_dir = args.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())
    submission = request_json(
        args.base_url,
        "/prompt",
        {"prompt": prompt, "client_id": client_id},
    )
    atomic_json_write(evidence_dir / "prompt-response.json", submission)
    prompt_id = submission["prompt_id"]

    poll_count = 0
    started = time.monotonic()
    while True:
        poll_count += 1
        response = request_json(args.base_url, f"/history/{prompt_id}")
        atomic_json_write(evidence_dir / "history-latest.json", response)
        if prompt_id in response:
            history = response[prompt_id]
            status = history.get("status", {})
            atomic_json_write(evidence_dir / "history-final.json", response)
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise RuntimeError(f"ComfyUI workflow failed: {status}")
            downloaded = download_outputs(args.base_url, history, evidence_dir / "downloaded")
            atomic_json_write(
                evidence_dir / "api-run.json",
                {
                    "status": "pass",
                    "client_id": client_id,
                    "prompt_id": prompt_id,
                    "poll_count": poll_count,
                    "elapsed_seconds": time.monotonic() - started,
                    "downloaded": downloaded,
                },
            )
            print(json.dumps({"prompt_id": prompt_id, "downloaded": downloaded}))
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
