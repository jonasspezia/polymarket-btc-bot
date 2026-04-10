from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EXPECTED_PROCESS_MARKERS = (
    "src.execution.engine",
    "05_validation_preflight.py",
)


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _iter_process_cmdlines() -> list[str]:
    cmdlines: list[str] = []
    for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            raw = cmdline_path.read_bytes()
        except OSError:
            continue

        if not raw:
            continue

        cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        if cmdline:
            cmdlines.append(cmdline)

    return cmdlines


def main() -> int:
    model_path = Path(
        os.getenv("MODEL_TARGET_PATH", "/app/data/models/lgbm_btc_5m.txt")
    )
    if not model_path.exists():
        return fail(f"missing_model:{model_path}")

    cmdlines = _iter_process_cmdlines()
    if not cmdlines:
        return fail("proc_scan_empty")

    if not any(
        marker in cmdline
        for marker in EXPECTED_PROCESS_MARKERS
        for cmdline in cmdlines
    ):
        return fail("expected_process_missing")

    manifests_dir = Path("/app/data/artifacts/runs")
    manifests = sorted(
        manifests_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if manifests:
        try:
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return fail(f"manifest_parse_error:{exc}")
        if payload.get("status") in {"startup_failed", "task_failed"}:
            return fail(f"manifest_status:{payload.get('status')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
