"""Import-layer guard: a module may import only from strictly lower layers.

This is the whole point of the scaffolding in this phase — it turns the layering in
ARCHITECTURE.md into something that fails loudly as soon as a module violates it.

    python code/evaluation/check_layers.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]

LAYERS: dict[str, int] = {
    "errors": 0,
    "ai": 1,
    "money": 1, "config": 1,
    "models": 2,
    "fx": 3, "trace": 3, "usage": 3, "ai.schemas": 3, "ai.cache": 3,
    "dataset": 4, "ai.client": 4,
    "ai.messages": 5, "ai.images": 5,
    "ai.facts": 6,
    "evidence": 7,
    "resolve": 8,
    "recurrence": 9,
    "forecast": 10,
    "safeamount": 11,
    "eligibility": 12, "spending": 12, "plans": 13,
    "outputs": 13,
    "validate": 14, "rank": 14,
    "explain": 15,
    "predict": 16,
    "agent": 17,
}


def module_name(path: Path) -> str:
    rel = path.relative_to(PKG_ROOT / "bow").with_suffix("")
    parts = [p for p in rel.parts if p != "__init__"]
    return ".".join(parts)


def imported_bow_modules(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("bow"):
            found.add(node.module.removeprefix("bow").lstrip("."))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("bow."):
                    found.add(alias.name.removeprefix("bow."))
    return {f for f in found if f}


def main() -> int:
    problems: list[str] = []
    for path in sorted((PKG_ROOT / "bow").rglob("*.py")):
        name = module_name(path)
        if not name:
            continue
        if name not in LAYERS:
            problems.append(f"{name}: not declared in LAYERS")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for dep in imported_bow_modules(tree):
            if dep not in LAYERS:
                problems.append(f"{name}: imports undeclared module {dep!r}")
            elif LAYERS[dep] >= LAYERS[name]:
                problems.append(
                    f"{name} (L{LAYERS[name]}) imports {dep} (L{LAYERS[dep]}) "
                    "— must be a strictly lower layer"
                )
    if problems:
        print("LAYER VIOLATIONS:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"layers: ok ({sum(1 for _ in (PKG_ROOT / 'bow').rglob('*.py'))} modules checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
