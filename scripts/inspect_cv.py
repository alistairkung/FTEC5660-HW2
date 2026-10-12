"""Print one CV's intermediate pipeline results for development inspection."""

import argparse
import asyncio
import json
from pathlib import Path

from hw2 import build_agent, cv_text, load_env_file, load_mcp_tools


async def inspect_cv(path: Path) -> None:
    load_env_file()
    text = cv_text(path)
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    result = await agent.ainvoke(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", required=True, type=Path, help="CV PDF to inspect")
    args = parser.parse_args()
    if not args.cv.is_file():
        parser.error(f"CV file does not exist: {args.cv}")
    asyncio.run(inspect_cv(args.cv))


if __name__ == "__main__":
    main()
