"""
Debug script to preview how `build_pair_user_input` renders pairs in markdown and JSON.

It loads real issues from the iOS Excel file and prints both formats for a few pairs
so you can eyeball the prompts before sending them to the LLM.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List, Tuple

import pandas as pd

# Make sure labler/ is on sys.path so we can import tikets_preraratior without a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from labler.tikets_preraratior import build_pair_user_input

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    # Older Python or environments without reconfigure; fallback to default.
    pass


DATA_PATH = Path(__file__).resolve().parent.parent / "labler" / "docs" / "input" / "Chattti_Develop_pairs_dual_size2_ios_filtered.xlsx"


def load_pairs_from_excel(path: Path, sample_pairs: int = 3) -> List[Tuple[dict, dict, int]]:
    """
    Return a handful of issue pairs (issue1, issue2, dual_id) from the Excel file.
    We pick the first two issues within each Dual_ID group.
    """
    df = pd.read_excel(path)
    pairs: List[Tuple[dict, dict, int]] = []
    for dual_id, group in df.groupby("Dual_ID"):
        if len(group) < 2:
            continue
        issues = group.head(2).to_dict(orient="records")
        pairs.append((issues[0], issues[1], int(dual_id)))
        if len(pairs) >= sample_pairs:
            break
    return pairs


def preview_pairs(pairs: Iterable[Tuple[dict, dict, int]]) -> None:
    """Print markdown and JSON prompts for each pair."""
    for issue1, issue2, dual_id in pairs:
        print("\n" + "=" * 80)
        print(f"Pair Dual_ID={dual_id} | {issue1.get('key')} vs {issue2.get('key')}")
        print("-" * 80)
        md_prompt = build_pair_user_input(issue1, issue2, format_type="markdown")
        print("MARKDOWN PROMPT:\n")
        print(md_prompt)
        print("\n" + "-" * 80)
        json_prompt = build_pair_user_input(issue1, issue2, format_type="json")
        print("JSON PROMPT:\n")
        print(json_prompt)
        print("=" * 80 + "\n")


if __name__ == "__main__":
    pairs = load_pairs_from_excel(DATA_PATH, sample_pairs=3)
    if not pairs:
        raise SystemExit("No pairs found in the Excel file.")
    preview_pairs(pairs)
