#!/usr/bin/env python3
"""Fit all identifiable tropomyosin parameters in multifil_jax.

This is a convenience entry point for ``fit_joint_experiments.py``.  It
always runs its ``tm_all`` stage, which jointly fits the TM steady-state and
kinetic parameters against ``data/pca_force.csv`` and ``data/ktr.csv``.

Examples:
    python fit_tm_all.py --maxiter 8 --popsize 4
    python fit_tm_all.py --start-json fit_previous.json --maxiter 20 --popsize 8
"""

from __future__ import annotations

import sys

from fit_joint_experiments import main


if __name__ == "__main__":
    # The underlying script owns all data loading, solver checks, and JSON
    # output.  Keep its CLI options available while fixing the fit stage.
    sys.argv[1:1] = ["--stage", "tm_all"]
    main()
