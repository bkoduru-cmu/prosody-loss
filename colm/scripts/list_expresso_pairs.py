"""
List Expresso parallel same-text groups (for CKA). Verifies data paths.
Usage:
  python -m colm.scripts.list_expresso_pairs --root /path/to/expresso --max-groups 5
"""
from __future__ import annotations

import argparse
import os
import sys

# Allow running as script or -m from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from colm.data.expresso import build_expresso_parallel_groups
from colm.config import EXPRESSO_ROOT


def main():
    p = argparse.ArgumentParser(description="List Expresso parallel same-text groups")
    p.add_argument("--root", default=EXPRESSO_ROOT, help="Expresso dataset root")
    p.add_argument("--max-groups", type=int, default=10, help="Max groups to print")
    p.add_argument("--emphasis-only", action="store_true", help="Show only emphasis subset")
    p.add_argument("--min-styles", type=int, default=2)
    args = p.parse_args()

    parallel, emphasis = build_expresso_parallel_groups(
        root=args.root,
        min_styles_per_group=args.min_styles,
    )
    print(f"Total parallel groups (same sentence, multiple styles): {len(parallel)}")
    print(f"Emphasis subset groups: {len(emphasis)}")

    groups = emphasis if args.emphasis_only else parallel
    for i, g in enumerate(groups[: args.max_groups]):
        print(f"\n--- Group {i + 1} ---")
        print(f"  text: {g['text_raw'][:80]}...")
        for u in g["utterances"]:
            print(f"    {u['style']}: {u['path']}")


if __name__ == "__main__":
    main()
