from __future__ import annotations

import os
import sys
from importlib import import_module
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = REPO_ROOT / "pybullet-mcp-server"
LOG_PATH = REPO_ROOT / "data" / "pybullet_mcp.log"

if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


def _redirect_stderr_to_log():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(LOG_PATH, "a", encoding="utf-8")
    stderr_fd = os.dup(2)
    os.dup2(log_handle.fileno(), 2)
    return log_handle, stderr_fd


def _import_mcp_safely():
    stdout_fd = os.dup(1)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), 1)
            return import_module("src.server").mcp
    finally:
        os.dup2(stdout_fd, 1)
        os.close(stdout_fd)


_LOG_HANDLE, _STDERR_FD = _redirect_stderr_to_log()
mcp = _import_mcp_safely()


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
