"""Console-script entry points. Stubs at this stage — implemented in later tasks."""

from __future__ import annotations

import argparse
import logging

from es_script_agent.data import load_dataset
from es_script_agent.es_client import make_client
from es_script_agent.indices.load import setup_indices


def setup_indices_cmd() -> None:
    """Drop, recreate, and bulk-load the ``loans`` and ``users`` indices."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(prog="setup-indices")
    parser.add_argument(
        "--dataset",
        default="default",
        help="Registered dataset name (default: %(default)s).",
    )
    args = parser.parse_args()

    adapter = load_dataset(args.dataset)
    client = make_client()
    counts = setup_indices(client, adapter)

    for index_name, count in counts.items():
        print(f"{index_name}: {count} docs")


def baseline_cmd() -> None:
    raise NotImplementedError("baseline is not implemented yet (Task 12)")


def rl_loop_cmd() -> None:
    raise NotImplementedError("rl-loop is not implemented yet (Task 16)")


def eval_cmd() -> None:
    raise NotImplementedError("eval is not implemented yet (Task 13)")
