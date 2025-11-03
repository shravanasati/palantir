#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.semantic_search import verify_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify if the model is loaded.")

    args = parser.parse_args()
    match args.command:
        case "verify":
            verify_model()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
