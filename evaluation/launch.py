from __future__ import annotations

import shutil
import subprocess
import sys


def require_torchrun() -> str:
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        raise RuntimeError("torchrun was not found in PATH")
    return torchrun


def module_command(module: str, args: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *args]


def torchrun_module_command(
    module: str,
    nproc_per_node: int,
    args: list[str],
) -> list[str]:
    return [
        require_torchrun(),
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        "--module",
        module,
        *args,
    ]


def run(command: list[str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command).returncode
