"""Context management.

Deliberately re-exports only the leaf modules. ``manager`` imports
``store.base``, which in turn imports ``context.ledger`` — re-exporting
``ContextManager`` here would make that a genuine import cycle whenever
``ctxproxy.store`` is imported before ``ctxproxy.app`` (which is exactly what
the CLI does). Import it from its own module instead:

    from ctxproxy.context.manager import ContextManager
"""

from .budget import Budget, compute_budget
from .ledger import SessionLedger
from .protect import ProtectedSet, classify, safe_cut_points

__all__ = [
    "Budget",
    "compute_budget",
    "SessionLedger",
    "ProtectedSet",
    "classify",
    "safe_cut_points",
]
