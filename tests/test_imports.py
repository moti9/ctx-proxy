"""Import-order safety.

Import cycles are order-dependent: `ctxproxy.app` pulls modules in an order that
happens to work, while the CLI imports `ctxproxy.store` first and blows up. Each
module is imported in a fresh interpreter so no earlier import can mask a cycle.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

MODULES = [
    "ctxproxy",
    "ctxproxy.app",
    "ctxproxy.cli",
    "ctxproxy.config",
    "ctxproxy.routes",
    "ctxproxy.state",
    "ctxproxy.session",
    "ctxproxy.backends",
    "ctxproxy.context",
    "ctxproxy.context.manager",
    "ctxproxy.context.summarizer",
    "ctxproxy.store",
    "ctxproxy.store.file",
    "ctxproxy.tokens",
    "ctxproxy.translate",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_standalone(module: str):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"importing {module} first failed:\n{result.stderr}"
