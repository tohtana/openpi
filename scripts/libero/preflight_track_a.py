#!/usr/bin/env python3
"""Preflight checks for Track A LeRobot pi0 LIBERO evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


@dataclass
class PreflightConfig:
    task_suite: str
    policy_path: str
    mujoco_gl: str
    gpu_ids: list[int]
    artifacts_dir: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_command(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 60) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=timeout,
        )
        return True, out.strip()
    except Exception as exc:
        msg = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            msg = exc.output.strip() or msg
        return False, msg


def _discover_gpus() -> tuple[int, list[str], list[int]]:
    ok, out = _run_command(["nvidia-smi", "-L"])
    if not ok:
        return 0, [], []

    names: list[str] = []
    ids: list[int] = []
    for line in out.splitlines():
        m = re.match(r"GPU\s+(\d+):\s+([^\(]+)", line.strip())
        if not m:
            continue
        ids.append(int(m.group(1)))
        names.append(m.group(2).strip())
    return len(ids), names, ids


def _check_lerobot_import() -> tuple[bool, str]:
    return _run_command([sys.executable, "-c", "import lerobot; print('ok')"], timeout=30)


def _check_lerobot_eval_help() -> tuple[bool, str]:
    return _run_command(["lerobot-eval", "--help"], timeout=30)


def _policy_resolve_ok(policy_path: str) -> bool:
    p = Path(policy_path)
    if p.exists():
        return True
    # Accept remote IDs like org/name.
    return bool(re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", policy_path))


def _check_backend(backend: str) -> tuple[bool, str]:
    env = dict(os.environ)
    env["MUJOCO_GL"] = backend
    code = (
        "import builtins;"
        "import os;"
        "builtins.input=lambda prompt='': 'N';"
        "from libero.libero.envs import OffScreenRenderEnv;"
        "print('ok', os.environ.get('MUJOCO_GL', ''))"
    )
    ok, out = _run_command([sys.executable, "-c", code], env=env, timeout=45)
    return ok, out


def run_preflight(cfg: PreflightConfig) -> dict[str, Any]:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    gpu_count, gpu_names, visible_gpu_ids = _discover_gpus()
    lerobot_import_ok, lerobot_import_msg = _check_lerobot_import()
    lerobot_eval_help_ok, lerobot_eval_help_msg = _check_lerobot_eval_help()
    policy_ok = _policy_resolve_ok(cfg.policy_path)

    attempts: list[dict[str, Any]] = []
    backend_order = [cfg.mujoco_gl]
    if cfg.mujoco_gl == "egl":
        backend_order.append("glx")

    effective_backend: str | None = None
    for backend in backend_order:
        ok, msg = _check_backend(backend)
        attempts.append({"backend": backend, "ok": ok, "message": msg})
        if ok:
            effective_backend = backend
            break

    checks = {
        "gpu_ids_requested_present": all(gid in visible_gpu_ids for gid in cfg.gpu_ids),
        "lerobot_import": lerobot_import_ok,
        "lerobot_eval_help": lerobot_eval_help_ok,
        "policy_resolve": policy_ok,
        "offscreen_backend": effective_backend is not None,
    }

    config_payload = asdict(cfg)
    config_payload["artifacts_dir"] = str(cfg.artifacts_dir)

    report = {
        "schema_version": "1",
        "generated_at_utc": _utc_now(),
        "config": config_payload,
        "checks": checks,
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "display_env": os.environ.get("DISPLAY", ""),
        "mujoco_gl_requested": cfg.mujoco_gl,
        "mujoco_gl_effective": effective_backend,
        "backend_attempts": attempts,
        "python_version": sys.version,
        "lerobot_import_ok": lerobot_import_ok,
        "lerobot_import_message": lerobot_import_msg,
        "lerobot_eval_help_ok": lerobot_eval_help_ok,
        "lerobot_eval_help_message": lerobot_eval_help_msg,
        "policy_resolve_ok": policy_ok,
    }
    return report


def write_preflight_report(report: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_gpu_ids(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-suite", required=True)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--mujoco-gl", required=True, choices=["egl", "glx"])
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU ids")
    parser.add_argument("--artifacts-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cfg = PreflightConfig(
        task_suite=args.task_suite,
        policy_path=args.policy_path,
        mujoco_gl=args.mujoco_gl,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        artifacts_dir=args.artifacts_dir,
    )
    report = run_preflight(cfg)
    out_path = cfg.artifacts_dir / "preflight.json"
    write_preflight_report(report, out_path)

    required_ok = (
        report["checks"].get("gpu_ids_requested_present", False)
        and report.get("lerobot_import_ok", False)
        and report.get("lerobot_eval_help_ok", False)
        and report.get("policy_resolve_ok", False)
        and report["checks"].get("offscreen_backend", False)
    )
    return 0 if required_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
