"""Checkpointing for the trajectories, at frame granularity.

A frame and its checkpoint are written together, so an interrupted run resumes
at exactly the right place rather than somewhere near it.
"""

import os

import numpy as np


def save(path, atoms, reference, equil_done, n_frames, motif):
    """Write atomically: an interruption mid-write must not be able to leave a
    truncated checkpoint behind."""
    path = str(path)
    np.savez(path + ".tmp.npz",
             positions=atoms.get_positions(), momenta=atoms.get_momenta(),
             reference=reference, equil_done=equil_done, n_frames=n_frames,
             motif=motif)
    os.replace(path + ".tmp.npz", path)


def load(path, motif):
    """Read a checkpoint and copy the arrays out.

    np.load returns a lazy NpzFile that holds the file open, and on Windows
    that blocks the atomic replace when the next checkpoint is written - hence
    the context manager and the copies.
    """
    with np.load(path, allow_pickle=True) as stored:
        state = {
            "motif": str(stored["motif"]),
            "positions": stored["positions"].copy(),
            "momenta": stored["momenta"].copy(),
            "reference": stored["reference"].copy(),
            "equil_done": int(stored["equil_done"]),
            "n_frames": int(stored["n_frames"]),
        }
    if state["motif"] != motif:
        raise SystemExit(f"{path} is for motif {state['motif']}, not {motif}; "
                         f"delete it")
    return state
