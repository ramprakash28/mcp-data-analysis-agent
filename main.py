"""
main.py — CLI entry point for the MCP Data Analysis Agent
==========================================================

Usage
-----
  python main.py data/sample_sales.csv
  python main.py data/sample_sales.csv --goal "Focus on regional performance"
  python main.py data/sample_sales.csv --goal "Find top 5 products by revenue"
"""

import argparse
import asyncio
import sys
from pathlib import Path

from agent.agent import DataAnalysisAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous MCP-powered data analysis agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_path",
        help="Path to the CSV file you want to analyse",
    )
    parser.add_argument(
        "--goal",
        default="Perform a comprehensive exploratory data analysis and highlight key insights.",
        help="Natural-language analysis goal (default: comprehensive EDA)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[error] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  MCP Data Analysis Agent")
    print("=" * 60)

    agent  = DataAnalysisAgent()
    result = await agent.run(str(csv_path), args.goal)

    print("\n" + "=" * 60)
    print("  Agent Summary")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
