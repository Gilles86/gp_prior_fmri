"""Per-subject provenance manifest.

Each fitted subject gets a sibling ``_desc-manifest.json`` next to
its TSVs. Records git SHAs of gp_prior_fmri + braincoder + the
source-dataset project, all CLI args, voxel-selection rule, and run
timestamps. Future-you can stand on any TSV and reproduce it.
"""
from __future__ import annotations

import json
import os.path as op
import subprocess


def git_sha(path: str) -> str | None:
    """Short HEAD SHA of the repo containing ``path``, or None."""
    try:
        out = subprocess.check_output(
            ['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except Exception:
        return None


def write_manifest(output_dir: str, subject: str,
                    session: str | None, manifest: dict) -> str:
    """Write the manifest JSON. Returns the path."""
    suffix = f'_ses-{session}' if session is not None else ''
    fn = op.join(output_dir,
                 f'sub-{subject}{suffix}_desc-manifest.json')
    with open(fn, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    return fn
