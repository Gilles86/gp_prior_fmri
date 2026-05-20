"""Cortical surface + geodesic distance helpers.

Used by the fit pipeline to build the per-subject distance matrix
that indexes the GP prior. Adapter agnostic — needs only a masker
(for voxel centroids) and the subject's white-matter GIfTI surface
paths.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from braincoder.utils.cortex import geodesic_distance_matrix


def voxel_centroids_mm(masker):
    """Scanner-space (mm) coordinates of the in-mask voxels."""
    mask_img = masker.mask_img_
    affine = mask_img.affine
    ijk = np.argwhere(mask_img.get_fdata().astype(bool))
    return nib.affines.apply_affine(affine, ijk).astype(np.float32)


def load_white_surface(sub, hemi_letter: str):
    """Load fmriprep's white-matter GIfTI surface for one hemisphere.

    Returns ``(vertices_mm, faces)`` in T1w space so coordinates match
    the EPI mask directly. ``sub`` is any object with a
    ``get_surf_info()`` method returning ``{hemi: {'inner': path}}``
    — both neural_priors' and tms_risk's Subject classes have it.
    """
    surf_info = sub.get_surf_info()
    gii = nib.load(surf_info[hemi_letter]['inner'])
    return (gii.darrays[0].data.astype(np.float32),
            gii.darrays[1].data.astype(np.int64))


def cortical_distance_matrix(xyz, vertices, faces, progressbar=True):
    """Project voxel centroids to nearest surface vertex, return
    pairwise geodesic distances (in mm).

    Returns ``(D, vtx_idx, snap_dist)``: ``D[i, j]`` = geodesic
    distance between voxels i and j; ``vtx_idx`` matches each voxel
    to a vertex; ``snap_dist`` is the voxel→vertex Euclidean distance
    (sanity check; >5 mm suggests coregistration trouble).
    """
    tree = cKDTree(vertices)
    snap_dist, vtx_idx = tree.query(xyz, k=1)
    vtx_idx = np.asarray(vtx_idx, dtype=int)
    snap_dist = np.asarray(snap_dist, dtype=np.float32)

    # Multiple voxels can project to the same vertex (folded cortex,
    # 3-D voxels). Compute distances on unique vertices then expand.
    unique_vtx, inverse = np.unique(vtx_idx, return_inverse=True)
    D_unique = geodesic_distance_matrix(
        vertices, faces, source_indices=unique_vtx,
        progressbar=progressbar)
    D = D_unique[np.ix_(inverse, inverse)].astype(np.float32)
    np.fill_diagonal(D, 0.0)
    return D, vtx_idx, snap_dist


def roi_to_hemi_letter(roi: str) -> str:
    """``NPCr → 'R'``, ``NPCl → 'L'``. Raise on bilateral / ambiguous."""
    if roi.endswith(('r', 'R')):
        return 'R'
    if roi.endswith(('l', 'L')):
        return 'L'
    raise ValueError(
        f"Can't pick a hemisphere from roi={roi!r}. Use NPCl/NPCr; "
        f"for bilateral, run twice.")
