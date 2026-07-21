"""
probe_variability.py
Probe placement variability analysis for NeuroCaptain 2.0.

Two-button workflow:
  1. Export Subject JSON — saves per-subject optode metrics
  2. Run Group Analysis  — loads a directory of subject JSONs, computes
     inter-subject variability, and visualizes results in the viewport.

Group analysis always uses head-frame-normalized (nm_scaled) coordinates:
each optode is projected into a per-subject fiducial frame (origin at Cz)
and divided by anatomical scale factors (|Lpa-Rpa|, |Nz-Iz|, 2*|mid-Cz|).
This handles both coordinate alignment and head-size normalization.
SD is converted back to mm via mean scale factors for reporting.

Reference: AtlasViewer plotProbePlacementVariation.m (BUNPC)
"""

import bpy
import json
import csv
import math
import os
import numpy as np
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Vector
from mathutils.bvhtree import BVHTree


# ── Constants ────────────────────────────────────────────────────────────────

FORMAT_VERSION = "2.0"
COORDSYS = "neuromag"
SS_THRESH = 15.0
COLLECTION_NAME = "Variability_Analysis"
N_CIRCLE_VERTS = 24


# ── Landmark label lists (vertex-index order from brain1020 output) ──────────

LABELS_1010 = [
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz",
    "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1",
    "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2",
    "FC1", "FC3", "FC5", "FC2", "FC4", "FC6",
    "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4",
    "CP1", "CP3", "CP5", "CP2", "CP4", "CP6",
    "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4",
    "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9",
    "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10",
]

LABELS_1020 = [
    "Nz", "Iz", "Lpa", "Rpa", "Cz",
    "T3", "C3", "Cz", "C4", "T4",
    "Fpz", "Fz", "Cz", "Pz", "Oz",
    "F7", "Fp1", "T5", "O1", "F8", "Fp2", "T6", "O2",
    "F3", "F4", "P3", "P4",
]


# ═══════════════════════════════════════════════════════════════════════════════
# PURE-PYTHON 3-D VECTOR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sub(a, b):   return [a[i] - b[i] for i in range(3)]
def _add(a, b):   return [a[i] + b[i] for i in range(3)]
def _scale(a, s): return [a[i] * s    for i in range(3)]
def _dot(a, b):   return sum(a[i] * b[i] for i in range(3))
def _cross(a, b): return [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
]
def _norm(a):
    n = math.sqrt(_dot(a, a))
    return _scale(a, 1.0 / n) if n > 1e-12 else [0.0, 0.0, 0.0]

def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i])**2 for i in range(3)))


# ═══════════════════════════════════════════════════════════════════════════════
# HEAD COORDINATE FRAME (Neuromag-style RAS)
# ═══════════════════════════════════════════════════════════════════════════════

def build_head_frame(lm):
    """
    Construct a Neuromag-style head frame from 5 fiducials.

    Origin: midpoint(LPA, RPA)
    Z: normalize(Cz - origin)                          -> toward vertex
    X: normalize(Nz - origin), orthogonalized vs Z     -> toward anterior
    Y: cross(Z, X)                                      -> toward left

    Scale factors:
        scale_x = |Nz  - Iz|           (AP span)
        scale_y = |Lpa - Rpa|          (lateral span)
        scale_z = |origin - Cz|        (inferior-superior span)

    Returns dict with origin, x/y/z axis vectors, and scale factors,
    or None if any required landmark is missing.
    """
    cz  = lm.get("Cz")
    nz  = lm.get("Nz")
    iz  = lm.get("Iz")
    lpa = lm.get("Lpa") or lm.get("LPA")
    rpa = lm.get("Rpa") or lm.get("RPA")

    missing = [k for k, v in {"Cz": cz, "Nz": nz, "Iz": iz,
                               "Lpa": lpa, "Rpa": rpa}.items() if not v]
    if missing:
        return None

    origin = [(lpa[i] + rpa[i]) / 2.0 for i in range(3)]

    z = _norm(_sub(cz, origin))

    x_raw = _norm(_sub(nz, origin))
    x = _norm(_sub(x_raw, _scale(z, _dot(x_raw, z))))

    y = _norm(_cross(z, x))

    scale_x = _dist(nz, iz)
    scale_y = _dist(lpa, rpa)
    scale_z = _dist(origin, cz)

    return {
        "origin":  origin,
        "x":       x,
        "y":       y,
        "z":       z,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "scale_z": scale_z,
    }


def project_into_head_frame(pos, frame):
    """
    Project a world position into the head frame.
    Returns (x_mm, y_mm, z_mm, x_scaled, y_scaled, z_scaled).
    """
    d = _sub(pos, frame["origin"])
    fx = _dot(d, frame["x"])
    fy = _dot(d, frame["y"])
    fz = _dot(d, frame["z"])

    sx = frame["scale_x"]
    sy = frame["scale_y"]
    sz = frame["scale_z"]

    fx_sc = fx / sx if sx > 1e-6 else 0.0
    fy_sc = fy / sy if sy > 1e-6 else 0.0
    fz_sc = fz / sz if sz > 1e-6 else 0.0

    return fx, fy, fz, fx_sc, fy_sc, fz_sc


# ═══════════════════════════════════════════════════════════════════════════════
# SCENE DATA EXTRACTION (for Button 1)
# ═══════════════════════════════════════════════════════════════════════════════

def get_landmarks():
    """Read landmark positions from LandmarkMesh. Returns {label: [x,y,z]}."""
    obj = bpy.data.objects.get("LandmarkMesh")
    if obj is None:
        return {}

    n = len(obj.data.vertices)
    labels = LABELS_1010 if n >= 77 else LABELS_1020
    out = {}

    for i, v in enumerate(obj.data.vertices):
        if i >= len(labels):
            break
        lbl = labels[i]
        if lbl and lbl not in out:
            wp = obj.matrix_world @ v.co
            out[lbl] = [wp.x, wp.y, wp.z]

    return out


def get_optodes():
    """
    Collect and sort Source_N / Detector_N objects.
    Returns [(obj, type_str, index_int), ...] sorted sources-first.
    """
    found = []
    for obj in bpy.data.objects:
        parts = obj.name.split("_")
        if len(parts) < 2:
            continue
        kind = parts[0]
        if kind not in ("Source", "Detector"):
            continue
        try:
            idx = int(parts[1])
        except ValueError:
            continue
        found.append((obj, kind.lower(), idx))

    found.sort(key=lambda t: (0 if t[1] == "source" else 1, t[2]))
    return found


def compute_short_sep_flags(optodes_with_positions, threshold_mm):
    """
    Tag short-separation optodes.

    Only source-detector pairs are checked.  When a source and detector are
    within *threshold_mm*, only the detector is flagged — the source still
    participates in regular-separation channels and must not be excluded.
    """
    names = [t[0] for t in optodes_with_positions]
    poses = [t[1] for t in optodes_with_positions]
    flags = {n: False for n in names}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            is_sd_pair = (names[i].startswith("Source") and names[j].startswith("Detector")) or \
                         (names[j].startswith("Source") and names[i].startswith("Detector"))
            if not is_sd_pair:
                continue
            if _dist(poses[i], poses[j]) < threshold_mm:
                if names[i].startswith("Detector"):
                    flags[names[i]] = True
                else:
                    flags[names[j]] = True

    return flags


def bary_snap(pos, lm_obj, depsgraph):
    """
    Snap a world position to the nearest point on LandmarkMesh.
    Returns (face_idx, w1, w2, w3, snap_x, snap_y, snap_z, snap_dist_mm,
             vertex_labels).
    vertex_labels is a list of 3 label strings for the triangle vertices.
    """
    from .landmark_labels import get_landmark_labels

    bvh = BVHTree.FromObject(lm_obj, depsgraph)
    loc, _normal, face_idx, _dist_val = bvh.find_nearest(Vector(pos))

    if loc is None:
        return 0, 0.333, 0.333, 0.334, pos[0], pos[1], pos[2], 0.0, []

    poly = lm_obj.data.polygons[face_idx]
    vert_indices = list(poly.vertices[:3])
    verts = [list(lm_obj.matrix_world @ lm_obj.data.vertices[vi].co)
             for vi in vert_indices]
    v0, v1, v2 = verts
    p = list(loc)

    all_labels = get_landmark_labels(lm_obj)
    vlabels = [all_labels[vi] if vi < len(all_labels) else f"v{vi}"
               for vi in vert_indices]

    def tri_area(a, b, c):
        ab = _sub(b, a)
        ac = _sub(c, a)
        cr = _cross(ab, ac)
        return math.sqrt(_dot(cr, cr)) / 2.0

    A = tri_area(v0, v1, v2)
    if A < 1e-12:
        w1, w2, w3 = 0.333, 0.333, 0.334
    else:
        w1 = tri_area(p, v1, v2) / A
        w2 = tri_area(v0, p, v2) / A
        w3 = tri_area(v0, v1, p) / A

    snap_dist = _dist(pos, p)
    return face_idx, round(w1, 6), round(w2, 6), round(w3, 6), \
           loc.x, loc.y, loc.z, snap_dist, vlabels


def build_subject_data(context, subject_id):
    """
    Extract all optode metrics from the current Blender scene.
    Returns a dict ready for JSON serialization, or raises RuntimeError.
    """
    depsgraph = context.evaluated_depsgraph_get()

    lm = get_landmarks()
    if not lm:
        raise RuntimeError("No LandmarkMesh found in scene.")

    frame = build_head_frame(lm)
    if frame is None:
        raise RuntimeError(
            "Cannot build head frame — missing one or more fiducials "
            "(Nz, Iz, Lpa, Rpa, Cz) on LandmarkMesh."
        )

    raw_optodes = get_optodes()
    if not raw_optodes:
        raise RuntimeError("No Source_* or Detector_* objects found.")

    lm_obj = bpy.data.objects.get("LandmarkMesh")

    opt_name_pos = []
    for obj, kind, idx in raw_optodes:
        ev = obj.evaluated_get(depsgraph)
        pos = list(ev.matrix_world.translation)
        opt_name_pos.append((obj.name, pos))

    ss_flags = compute_short_sep_flags(opt_name_pos, SS_THRESH)

    fiducial_names = ["Nz", "Iz", "Lpa", "Rpa", "Cz"]
    optodes_out = []

    for obj, kind, idx in raw_optodes:
        ev = obj.evaluated_get(depsgraph)
        pos = list(ev.matrix_world.translation)

        nm_x, nm_y, nm_z, nm_xs, nm_ys, nm_zs = project_into_head_frame(pos, frame)

        bary_data = {}
        if lm_obj:
            fi, bw1, bw2, bw3, sx, sy, sz, sdist, vlabels = bary_snap(pos, lm_obj, depsgraph)
            bary_data = {
                "face_idx": fi,
                "weights": [bw1, bw2, bw3],
                "vertex_labels": vlabels,
                "snap_pos": [round(sx, 6), round(sy, 6), round(sz, 6)],
                "snap_dist_mm": round(sdist, 4),
            }

        fid_dists = {}
        for fid in fiducial_names:
            if fid in lm:
                fid_dists[fid] = round(_dist(pos, lm[fid]), 3)

        optodes_out.append({
            "name": obj.name,
            "type": kind,
            "index": idx,
            "is_anchor": bool(obj.get("is_anchor", False)),
            "is_short_sep": ss_flags.get(obj.name, False),
            "world_pos": [round(pos[0], 6), round(pos[1], 6), round(pos[2], 6)],
            "hf_mm": [round(nm_x, 6), round(nm_y, 6), round(nm_z, 6)],
            "hf_scaled": [round(nm_xs, 6), round(nm_ys, 6), round(nm_zs, 6)],
            "bary": bary_data,
            "fiducial_distances_mm": fid_dists,
        })

    landmarks_out = {}
    for k, v in lm.items():
        landmarks_out[k] = [round(c, 6) for c in v]

    return {
        "format_version": FORMAT_VERSION,
        "coordsys": COORDSYS,
        "subject_id": subject_id,
        "ss_threshold_mm": SS_THRESH,
        "landmarks": landmarks_out,
        "head_frame": {
            "origin": [round(c, 6) for c in frame["origin"]],
            "x_axis": [round(c, 6) for c in frame["x"]],
            "y_axis": [round(c, 6) for c in frame["y"]],
            "z_axis": [round(c, 6) for c in frame["z"]],
            "scale_x_mm": round(frame["scale_x"], 6),
            "scale_y_mm": round(frame["scale_y"], 6),
            "scale_z_mm": round(frame["scale_z"], 6),
        },
        "optodes": optodes_out,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP ANALYSIS (for Button 2)
# ═══════════════════════════════════════════════════════════════════════════════

def load_subject_json(filepath):
    """Load and validate a subject JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{os.path.basename(filepath)}: unsupported format_version "
            f"'{data.get('format_version')}' (expected '{FORMAT_VERSION}')"
        )
    if "optodes" not in data or not data["optodes"]:
        raise ValueError(f"{os.path.basename(filepath)}: no optodes found")

    return data


def load_subject_directory(dirpath):
    """Scan a directory for valid subject JSON files. Returns list of dicts."""
    subjects = []
    for fname in sorted(os.listdir(dirpath)):
        if not fname.lower().endswith(".json"):
            continue
        fpath = os.path.join(dirpath, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            subj = load_subject_json(fpath)
            subjects.append(subj)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  Skipping {fname}: {e}")
    return subjects


def align_subjects(subjects, exclude_short_sep=True):
    """
    Match optodes by name across subjects using inner join.

    If *exclude_short_sep* is True, any optode flagged as short-separation
    in any subject is excluded (consistent with AtlasViewer methodology).

    Returns:
        optode_names: list of str
        nm_mm:       np.ndarray [N_optodes, 3, N_subjects]
        nm_scaled:   np.ndarray [N_optodes, 3, N_subjects]
        world_pos:    np.ndarray [N_optodes, 3, N_subjects]
        subject_ids:  list of str
        scales:       np.ndarray [N_subjects, 3]  (scale_x, scale_y, scale_z per subject)
    """
    name_sets = []
    for subj in subjects:
        names = {opt["name"] for opt in subj["optodes"]}
        name_sets.append(names)

    common_names = name_sets[0]
    for ns in name_sets[1:]:
        common_names = common_names & ns

    if exclude_short_sep:
        ss_names = set()
        for subj in subjects:
            for opt in subj["optodes"]:
                if opt.get("is_short_sep", False) and opt.get("type") == "detector":
                    ss_names.add(opt["name"])
        if ss_names:
            common_names = common_names - ss_names
            print(f"  Excluded {len(ss_names)} short-separation detector(s): "
                  f"{sorted(ss_names)}")

    ref_optodes = subjects[0]["optodes"]
    ordered_names = [o["name"] for o in ref_optodes if o["name"] in common_names]

    n_opt = len(ordered_names)
    n_subj = len(subjects)
    nm_mm = np.zeros((n_opt, 3, n_subj))
    nm_scaled = np.zeros((n_opt, 3, n_subj))
    world_pos = np.zeros((n_opt, 3, n_subj))
    scales = np.zeros((n_subj, 3))
    subject_ids = []

    for s, subj in enumerate(subjects):
        subject_ids.append(subj.get("subject_id", f"subject_{s}"))
        fr = subj.get("head_frame", subj.get("neuromag_frame", {}))
        scales[s, 0] = fr.get("scale_x_mm", 1.0)
        scales[s, 1] = fr.get("scale_y_mm", 1.0)
        scales[s, 2] = fr.get("scale_z_mm", 1.0)
        lookup = {o["name"]: o for o in subj["optodes"]}
        for i, name in enumerate(ordered_names):
            opt = lookup[name]
            nm_mm[i, :, s] = opt.get("hf_mm", opt.get("neuromag_mm"))
            nm_scaled[i, :, s] = opt.get("hf_scaled", opt.get("neuromag_scaled"))
            world_pos[i, :, s] = opt["world_pos"]

    return ordered_names, nm_mm, nm_scaled, world_pos, subject_ids, scales


def compute_intersubject_stats(positions, axis_scales=None):
    """
    Compute inter-subject variability from a [N_optodes, 3, N_subjects] array.

    If *axis_scales* ([3] array of mm-per-unit) is provided, positions are
    in dimensionless scaled coordinates and all output values are converted
    to mm by multiplying per-axis differences by axis_scales.

    Returns dict:
        mean_pos:      [N_optodes, 3]            mean position (mm or scaled)
        sd:            [N_optodes]                population SD (mm)
        sd_per_axis:   [N_optodes, 3]            per-axis SD (mm)
        errors:        [N_optodes, N_subjects]   per-subject Euclidean error (mm)
    """
    n_opt, _, n_subj = positions.shape

    mean_pos = np.mean(positions, axis=2)

    errors = np.zeros((n_opt, n_subj))
    for s in range(n_subj):
        diff = positions[:, :, s] - mean_pos
        if axis_scales is not None:
            diff = diff * axis_scales
        errors[:, s] = np.sqrt(np.sum(diff ** 2, axis=1))

    sd = np.sqrt(np.sum(errors ** 2, axis=1) / n_subj)

    sd_per_axis = np.std(positions, axis=2, ddof=0)
    if axis_scales is not None:
        sd_per_axis = sd_per_axis * axis_scales

    mean_pos_out = mean_pos * axis_scales if axis_scales is not None else mean_pos

    return {
        "mean_pos": mean_pos_out,
        "sd": sd,
        "sd_per_axis": sd_per_axis,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# M1 / M2 / M3 METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_m1_landmark_error(subjects, optode_names):
    """
    M1: Landmark distance error.

    For each optode, take its distance to each of the 5 fiducials in every
    subject.  Compute the population SD of that distance across subjects,
    then average the 5 per-fiducial SDs into a single M1 value (mm).

    A high M1 means the optode's position relative to the 10-20 framework
    is inconsistent across registrations.
    """
    fid_names = ["Nz", "Iz", "Lpa", "Rpa", "Cz"]
    n_opt = len(optode_names)
    n_subj = len(subjects)

    fid_dists = np.full((n_opt, len(fid_names), n_subj), np.nan)

    for s, subj in enumerate(subjects):
        lookup = {o["name"]: o for o in subj["optodes"]}
        for i, name in enumerate(optode_names):
            opt = lookup.get(name)
            if opt is None:
                continue
            fd = opt.get("fiducial_distances_mm", {})
            for f, fid in enumerate(fid_names):
                if fid in fd:
                    fid_dists[i, f, s] = fd[fid]

    m1 = np.zeros(n_opt)
    for i in range(n_opt):
        sds = []
        for f in range(len(fid_names)):
            vals = fid_dists[i, f, :]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 2:
                sds.append(np.std(valid, ddof=0))
        m1[i] = np.mean(sds) if sds else 0.0

    return m1


def compute_m2_interoptode_distance(world_pos, optode_names):
    """
    M2: Interoptode distance preservation.

    For every unique optode pair, compute their Euclidean separation in
    each subject.  The population SD of that separation across subjects
    measures how well the pair's geometry is preserved.

    M2 per optode = mean pair-SD over all pairs that include it.
    """
    n_opt, _, n_subj = world_pos.shape

    if n_opt < 2 or n_subj < 2:
        return np.zeros(n_opt)

    n_pairs = n_opt * (n_opt - 1) // 2
    pair_dists = np.zeros((n_pairs, n_subj))
    pair_indices = []

    p = 0
    for i in range(n_opt):
        for j in range(i + 1, n_opt):
            for s in range(n_subj):
                pair_dists[p, s] = np.linalg.norm(
                    world_pos[i, :, s] - world_pos[j, :, s])
            pair_indices.append((i, j))
            p += 1

    pair_sd = np.std(pair_dists, axis=1, ddof=0)

    m2 = np.zeros(n_opt)
    for i in range(n_opt):
        relevant = [pair_sd[p] for p, (a, b) in enumerate(pair_indices)
                     if a == i or b == i]
        m2[i] = np.mean(relevant) if relevant else 0.0

    return m2


def compute_m3_barycentric_stability(subjects, optode_names):
    """
    M3: Barycentric stability.

    For each optode, collect the barycentric snap position (nearest point
    on LandmarkMesh) across subjects, project into each subject's
    head coordinate frame with anatomical normalization, then compute
    population SD converted back to mm via mean scale factors.
    """
    n_opt = len(optode_names)
    n_subj = len(subjects)

    snap_scaled = np.full((n_opt, 3, n_subj), np.nan)
    scales = np.zeros((n_subj, 3))

    for s, subj in enumerate(subjects):
        fr = subj.get("head_frame", subj.get("neuromag_frame", {}))
        frame = {
            "origin":  fr.get("origin", [0, 0, 0]),
            "x":       fr.get("x_axis", [1, 0, 0]),
            "y":       fr.get("y_axis", [0, 1, 0]),
            "z":       fr.get("z_axis", [0, 0, 1]),
            "scale_x": fr.get("scale_x_mm", 1.0),
            "scale_y": fr.get("scale_y_mm", 1.0),
            "scale_z": fr.get("scale_z_mm", 1.0),
        }
        scales[s] = [frame["scale_x"], frame["scale_y"], frame["scale_z"]]

        lookup = {o["name"]: o for o in subj["optodes"]}
        for i, name in enumerate(optode_names):
            opt = lookup.get(name)
            if opt is None:
                continue
            bary = opt.get("bary", {})
            sp = bary.get("snap_pos")
            if sp is not None:
                _, _, _, xs, ys, zs = project_into_head_frame(sp, frame)
                snap_scaled[i, :, s] = [xs, ys, zs]

    mean_scales = np.mean(scales, axis=0)

    m3 = np.zeros(n_opt)
    for i in range(n_opt):
        valid_mask = ~np.isnan(snap_scaled[i, 0, :])
        n_valid = int(np.sum(valid_mask))
        if n_valid < 2:
            continue
        coords = snap_scaled[i, :, valid_mask]
        if coords.shape[0] == 3:
            coords = coords.T
        mean_sp = np.mean(coords, axis=0)
        diff_mm = (coords - mean_sp) * mean_scales
        errors = np.sqrt(np.sum(diff_mm ** 2, axis=1))
        m3[i] = np.sqrt(np.sum(errors ** 2) / n_valid)

    return m3


_BARY_MINOR_WEIGHT = 0.05

def _nearest3_bary(optode_pos, landmark_dict):
    """
    Find the 3 nearest landmarks to optode_pos, compute barycentric weights.
    Returns (full_label_triplet, full_weights, sig_label_tuple, sig_weights).

    sig_label_tuple contains only the labels whose weight >= _BARY_MINOR_WEIGHT,
    used for triangle comparison so that near-zero vertices don't cause
    spurious disagreement.
    """
    if len(landmark_dict) < 3:
        return None, None, None, None

    dists = []
    for label, lpos in landmark_dict.items():
        dists.append((_dist(optode_pos, lpos), label, lpos))
    dists.sort(key=lambda x: x[0])

    nearest = dists[:3]
    labels_unsorted = [n[1] for n in nearest]
    verts = [n[2] for n in nearest]

    v0, v1, v2 = verts
    def tri_area(a, b, c):
        ab = _sub(b, a)
        ac = _sub(c, a)
        cr = _cross(ab, ac)
        return math.sqrt(_dot(cr, cr)) / 2.0

    A = tri_area(v0, v1, v2)
    if A < 1e-12:
        w = [0.333, 0.333, 0.334]
    else:
        w = [
            tri_area(optode_pos, v1, v2) / A,
            tri_area(v0, optode_pos, v2) / A,
            tri_area(v0, v1, optode_pos) / A,
        ]

    sort_idx = sorted(range(3), key=lambda k: labels_unsorted[k])
    sorted_labels = tuple(labels_unsorted[k] for k in sort_idx)
    sorted_weights = [w[k] for k in sort_idx]

    sig_pairs = [(sorted_labels[k], sorted_weights[k])
                 for k in range(3) if sorted_weights[k] >= _BARY_MINOR_WEIGHT]
    sig_labels = tuple(p[0] for p in sig_pairs)
    sig_weights = [p[1] for p in sig_pairs]

    return sorted_labels, sorted_weights, sig_labels, sig_weights


def _find_probe_json(subject_dir):
    """Search for probe_config*.json, preferring the subject directory."""
    import glob as _glob
    # Search subject dir first, then parent, then siblings
    local = sorted(_glob.glob(os.path.join(subject_dir, "probe_config*.json")))
    if local:
        return local[-1]

    parent = os.path.dirname(subject_dir.rstrip(os.sep))
    if parent:
        parent_matches = sorted(_glob.glob(os.path.join(parent, "probe_config*.json")))
        if parent_matches:
            return parent_matches[-1]

        for entry in os.listdir(parent):
            full = os.path.join(parent, entry)
            if os.path.isdir(full) and full != subject_dir.rstrip(os.sep):
                sibling = sorted(_glob.glob(os.path.join(full, "probe_config*.json")))
                if sibling:
                    return sibling[-1]

    return None


def compute_m4_barycentric_consistency(subjects, optode_names,
                                       probe_json_path=None):
    """
    Barycentric consistency.

    Uses the registration vertex labels and weights from the probe JSON
    to reconstruct each optode's barycentric position on each subject's
    LandmarkMesh.  Compares the reconstructed position to the actual
    final position (world_pos) — the displacement is the cloth-sim drift.

    Returns per-optode arrays:
      m4_bary_displacement_mm — RMS displacement between barycentric-
                                predicted position and actual position
                                across subjects (mm).  Anchors should
                                be near zero; non-anchors reflect cloth
                                sim drift.
      m4_displacement_sd_mm  — SD of per-subject displacement across
                                subjects (mm).  Low = consistent drift,
                                high = variable drift.
      m4_max_displacement_mm — worst single-subject displacement (mm).

    Falls back to nearest-3-landmark method if probe JSON is unavailable.
    """
    n_opt = len(optode_names)
    n_subj = len(subjects)

    probe_reg = {}
    if probe_json_path and os.path.isfile(probe_json_path):
        with open(probe_json_path, "r") as f:
            probe_data = json.load(f)
        for opt in probe_data.get("optodes", []):
            reg = opt.get("registration", {})
            vindices = reg.get("vertex_indices", [])
            bcoords = reg.get("barycentric_coords", [])
            if vindices and len(vindices) == 3 and bcoords and len(bcoords) == 3:
                probe_reg[opt["name"]] = {
                    "indices": vindices,
                    "weights": bcoords,
                }

    n_lm_labels = len(LABELS_1010)

    m4_rms = np.zeros(n_opt)
    m4_sd = np.zeros(n_opt)
    m4_max = np.zeros(n_opt)
    m4_weight_sd = np.zeros(n_opt)

    for i, name in enumerate(optode_names):
        reg = probe_reg.get(name)
        if reg is None:
            continue

        reg_indices = reg["indices"]
        reg_weights = reg["weights"]

        subj_labels = []
        for idx in reg_indices:
            if idx < n_lm_labels and LABELS_1010[idx]:
                subj_labels.append(LABELS_1010[idx])
            else:
                subj_labels.append(None)
        if any(l is None for l in subj_labels):
            continue

        displacements = []
        all_weights = []

        for subj in subjects:
            lm = subj.get("landmarks", {})
            verts = [lm.get(l) for l in subj_labels]
            if any(v is None for v in verts):
                continue

            lookup = {o["name"]: o for o in subj["optodes"]}
            opt = lookup.get(name)
            if opt is None:
                continue
            actual_pos = opt.get("world_pos")
            if actual_pos is None:
                continue

            predicted = [0.0, 0.0, 0.0]
            for w, v in zip(reg_weights, verts):
                predicted[0] += w * v[0]
                predicted[1] += w * v[1]
                predicted[2] += w * v[2]

            disp = _dist(predicted, actual_pos)
            displacements.append(disp)

            v0 = np.array(verts[0])
            v1 = np.array(verts[1])
            v2 = np.array(verts[2])
            p = np.array(actual_pos)
            e0 = v1 - v0
            e1 = v2 - v0
            e2 = p - v0
            d00 = np.dot(e0, e0)
            d01 = np.dot(e0, e1)
            d11 = np.dot(e1, e1)
            d20 = np.dot(e2, e0)
            d21 = np.dot(e2, e1)
            denom = d00 * d11 - d01 * d01
            if abs(denom) > 1e-12:
                bv = (d11 * d20 - d01 * d21) / denom
                bw = (d00 * d21 - d01 * d20) / denom
                bu = 1.0 - bv - bw
                all_weights.append([bu, bv, bw])

        if len(displacements) >= 2:
            arr = np.array(displacements)
            m4_rms[i] = float(np.sqrt(np.mean(arr ** 2)))
            m4_sd[i] = float(np.std(arr, ddof=0))
            m4_max[i] = float(np.max(arr))

        if len(all_weights) >= 2:
            warr = np.array(all_weights)
            per_weight_sd = np.std(warr, axis=0, ddof=0)
            m4_weight_sd[i] = float(np.mean(per_weight_sd))

    return m4_rms, m4_sd, m4_max, m4_weight_sd


def compute_edge_distances(subjects, optode_names, probe_json_path=None):
    """
    Compute per-edge inter-optode distances across subjects.

    For each connection in the probe JSON, measures the Euclidean distance
    between the two connected optodes in each subject's world_pos coordinates.
    Reports mean, SD, and deviation from the defined rest_length.

    Edge type classification:
      - "anchor"   — at least one endpoint is an anchor optode
      - "stiff"    — is_flexible=False, neither endpoint is anchor
      - "flexible" — is_flexible=True, neither endpoint is anchor
    """
    if not probe_json_path or not os.path.isfile(probe_json_path):
        return []

    with open(probe_json_path, "r") as f:
        probe_data = json.load(f)

    anchor_set = set()
    for opt in probe_data.get("optodes", []):
        if opt.get("is_anchor", False):
            anchor_set.add(opt["name"])

    optode_name_set = set(optode_names)
    results = []

    for cd in probe_data.get("connections", []):
        n1, n2 = cd["optode1"], cd["optode2"]
        if n1 not in optode_name_set or n2 not in optode_name_set:
            continue

        is_flexible = cd.get("is_flexible", False)
        fix_dist = cd.get("fix_distance", False)
        rest_length = float(cd.get("rest_length", 0))

        if n1 in anchor_set or n2 in anchor_set:
            edge_type = "anchor"
        elif is_flexible:
            edge_type = "flexible"
        elif fix_dist:
            edge_type = "stiff*"
        else:
            edge_type = "stiff"

        distances = []
        for subj in subjects:
            lookup = {o["name"]: o for o in subj["optodes"]}
            o1 = lookup.get(n1)
            o2 = lookup.get(n2)
            if o1 is None or o2 is None:
                continue
            wp1 = o1.get("world_pos")
            wp2 = o2.get("world_pos")
            if wp1 is None or wp2 is None:
                continue
            distances.append(_dist(wp1, wp2))

        if distances:
            arr = np.array(distances)
            results.append({
                "optode1": n1,
                "optode2": n2,
                "edge_type": edge_type,
                "rest_length": rest_length,
                "n_subjects": len(distances),
                "mean_dist": float(np.mean(arr)),
                "sd_dist": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "min_dist": float(np.min(arr)),
                "max_dist": float(np.max(arr)),
                "mean_deviation": float(np.mean(arr) - rest_length),
            })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def _colormap(v):
    """Blue -> cyan -> green -> yellow -> red heatmap, v in [0, 1]."""
    v = max(0.0, min(1.0, v))
    if v < 0.25:
        t = v / 0.25
        return (0.0, t, 1.0, 1.0)
    elif v < 0.5:
        t = (v - 0.25) / 0.25
        return (0.0, 1.0, 1.0 - t, 1.0)
    elif v < 0.75:
        t = (v - 0.5) / 0.25
        return (t, 1.0, 0.0, 1.0)
    else:
        t = (v - 0.75) / 0.25
        return (1.0, 1.0 - t, 0.0, 1.0)


def _get_or_create_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _link_to_collection(obj, coll):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _make_material(name, color):
    """Create or reuse a simple material with given RGBA color."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
    return mat


def _find_head_mesh():
    obj = bpy.data.objects.get("headmesh")
    if obj and obj.type == 'MESH':
        return obj
    for obj in bpy.data.objects:
        if "headmesh" in obj.name.lower() and obj.type == 'MESH':
            return obj
    return None


def create_ellipse_mesh(position, normal, semi_x, semi_y, name, color):
    """
    Create an ellipse mesh oriented perpendicular to the given normal.
    Semi-axes are in mm. Color is RGBA tuple.
    """
    z_up = Vector((0, 0, 1))
    n_vec = Vector(normal).normalized()

    angles = [2 * math.pi * i / N_CIRCLE_VERTS for i in range(N_CIRCLE_VERTS)]
    verts_local = []
    for a in angles:
        verts_local.append(Vector((
            semi_x * math.cos(a),
            semi_y * math.sin(a),
            0.0
        )))

    rot = z_up.rotation_difference(n_vec)
    pos_vec = Vector(position)

    verts_world = []
    for v in verts_local:
        v.rotate(rot)
        verts_world.append(pos_vec + v)

    faces = [list(range(N_CIRCLE_VERTS))]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([list(v) for v in verts_world], [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    mat = _make_material(f"Var_{name}", color)
    obj.data.materials.append(mat)

    return obj


def create_disc_mesh(position, normal, radius, name, color):
    """Create a circular disc (special case of ellipse)."""
    return create_ellipse_mesh(position, normal, radius, radius, name, color)


def _compute_tangent_plane_ellipse(mean_pos, normal, error_vecs_nm_mm):
    """
    Project error vectors onto the tangent plane at mean_pos and compute
    ellipse semi-axes (std of the two tangent components).

    error_vecs_nm_mm: [N_subjects, 3] in Neuromag mm (difference from mean)
    normal: surface normal at mean_pos

    Returns (semi_x, semi_y) in mm.
    """
    n_vec = np.array(normal, dtype=float)
    n_vec /= np.linalg.norm(n_vec) + 1e-12

    z_up = np.array([0, 0, 1], dtype=float)
    if abs(np.dot(n_vec, z_up)) > 0.99:
        z_up = np.array([1, 0, 0], dtype=float)
    t1 = np.cross(n_vec, z_up)
    t1 /= np.linalg.norm(t1) + 1e-12
    t2 = np.cross(n_vec, t1)
    t2 /= np.linalg.norm(t2) + 1e-12

    proj1 = error_vecs_nm_mm @ t1
    proj2 = error_vecs_nm_mm @ t2

    semi_x = np.std(proj1, ddof=0)
    semi_y = np.std(proj2, ddof=0)

    return float(max(semi_x, 0.5)), float(max(semi_y, 0.5))


def build_variability_visualization(context, optode_names, optode_types,
                                     stats, mean_world_pos,
                                     show_per_subject, all_world_pos,
                                     subject_ids, nm_mm):
    """
    Create visualization meshes in the Variability_Analysis collection.

    mean_world_pos: [N_optodes, 3] — mean world positions for display
    all_world_pos:  [N_optodes, 3, N_subjects] — per-subject world positions
    """
    coll = _get_or_create_collection(COLLECTION_NAME)
    depsgraph = context.evaluated_depsgraph_get()

    head_mesh = _find_head_mesh()
    bvh = None
    if head_mesh:
        bvh = BVHTree.FromObject(head_mesh, depsgraph)

    sd_vals = stats["sd"]
    sd_min = float(np.min(sd_vals))
    sd_max = float(np.max(sd_vals))
    sd_range = sd_max - sd_min if sd_max > sd_min else 1.0

    n_opt = len(optode_names)
    n_subj = all_world_pos.shape[2]
    mean_pos = stats["mean_pos"]

    for i in range(n_opt):
        pos = mean_world_pos[i]
        pos_vec = Vector(pos)

        if bvh:
            _loc, normal, _fidx, _d = bvh.find_nearest(pos_vec)
            normal = list(normal) if normal else [0, 0, 1]
        else:
            normal = [0, 0, 1]

        error_vecs = all_world_pos[i, :, :].T - mean_pos[i]
        semi_x, semi_y = _compute_tangent_plane_ellipse(
            pos, normal, error_vecs
        )

        t = (sd_vals[i] - sd_min) / sd_range if sd_range > 0 else 0.5
        color = _colormap(t)

        ell_name = f"Var_{optode_names[i]}"
        ell = create_ellipse_mesh(pos, normal, semi_x, semi_y, ell_name, color)
        _link_to_collection(ell, coll)

    if show_per_subject and n_subj > 0:
        palette = _generate_subject_palette(n_subj)

        for s in range(n_subj):
            verts_all = []
            faces_all = []
            vert_offset = 0

            for i in range(n_opt):
                opt_pos = all_world_pos[i, :, s]
                pos_vec = Vector(opt_pos)

                if bvh:
                    _loc, normal, _fidx, _d = bvh.find_nearest(pos_vec)
                    normal = list(normal) if normal else [0, 0, 1]
                else:
                    normal = [0, 0, 1]

                disc_verts, disc_face = _disc_geometry(
                    opt_pos, normal, 1.5
                )
                for dv in disc_verts:
                    verts_all.append(dv)
                faces_all.append([vi + vert_offset for vi in disc_face])
                vert_offset += len(disc_verts)

            sid = subject_ids[s] if s < len(subject_ids) else f"subj_{s}"
            mesh_name = f"Var_Subject_{sid}"
            mesh = bpy.data.meshes.new(mesh_name)
            mesh.from_pydata(verts_all, [], faces_all)
            mesh.update()

            obj = bpy.data.objects.new(mesh_name, mesh)
            mat = _make_material(f"VarSubj_{sid}", palette[s])
            obj.data.materials.append(mat)
            _link_to_collection(obj, coll)


def _disc_geometry(position, normal, radius):
    """Generate vertices and a single face for a flat disc."""
    z_up = Vector((0, 0, 1))
    n_vec = Vector(normal).normalized()
    rot = z_up.rotation_difference(n_vec)
    pos_vec = Vector(position)

    verts = []
    for k in range(N_CIRCLE_VERTS):
        a = 2 * math.pi * k / N_CIRCLE_VERTS
        v = Vector((radius * math.cos(a), radius * math.sin(a), 0.0))
        v.rotate(rot)
        verts.append(list(pos_vec + v))

    face = list(range(N_CIRCLE_VERTS))
    return verts, face


def _generate_subject_palette(n):
    """Generate n visually distinct RGBA colors."""
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = _hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((r, g, b, 1.0))
    return colors


def _hsv_to_rgb(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return r, g, b


def export_summary_csv(filepath, optode_names, optode_type_map, stats,
                       n_subj, ellipse_axes, m1, m2, m3, world_pos=None,
                       m4_rms=None, m4_sd=None, m4_max=None, m4_wsd=None,
                       edge_dists=None):
    """Write comprehensive group variability summary to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["# NeuroCaptain — Head-Frame Normalized"])
        writer.writerow([
            "optode_name", "optode_type", "n_subjects",
            "mean_x_mm", "mean_y_mm", "mean_z_mm",
            "sd_mm", "sd_x_mm", "sd_y_mm", "sd_z_mm",
            "M1_landmark_error_mm", "M2_interoptode_sd_mm",
            "M3_bary_stability_mm",
            "M4_rms_mm", "M4_sd_mm", "M4_max_mm",
            "M4_weight_sd",
            "ellipse_semi_major_mm", "ellipse_semi_minor_mm",
        ])

        for i, name in enumerate(optode_names):
            mean = stats["mean_pos"][i]
            sd_ax = stats["sd_per_axis"][i]
            semi = ellipse_axes[i] if i < len(ellipse_axes) else (0, 0)
            m4r_val = round(m4_rms[i], 4) if m4_rms is not None else ""
            m4s_val = round(m4_sd[i], 4) if m4_sd is not None else ""
            m4m_val = round(m4_max[i], 4) if m4_max is not None else ""
            m4w_val = round(m4_wsd[i], 4) if m4_wsd is not None else ""
            writer.writerow([
                name,
                optode_type_map.get(name, ""),
                n_subj,
                round(mean[0], 4), round(mean[1], 4), round(mean[2], 4),
                round(stats["sd"][i], 4),
                round(sd_ax[0], 4), round(sd_ax[1], 4), round(sd_ax[2], 4),
                round(m1[i], 4), round(m2[i], 4), round(m3[i], 4),
                m4r_val, m4s_val, m4m_val, m4w_val,
                round(max(semi), 4), round(min(semi), 4),
            ])

        writer.writerow([])
        writer.writerow(["# NeuroCaptain Summary"])
        writer.writerow(["n_subjects", n_subj])
        writer.writerow(["n_optodes", len(optode_names)])
        writer.writerow(["coordinate_space", "head-frame normalized (nm_scaled)"])
        writer.writerow([])

        writer.writerow(["metric", "mean", "min", "max",
                         "min_optode", "max_optode"])

        sd = stats["sd"]
        metrics_list = [("SD_mm", sd),
                        ("M1_landmark_error_mm", m1),
                        ("M2_interoptode_sd_mm", m2),
                        ("M3_bary_stability_mm", m3)]
        if m4_rms is not None:
            metrics_list.extend([
                ("M4_rms_mm", m4_rms),
                ("M4_sd_mm", m4_sd),
                ("M4_max_mm", m4_max),
            ])
        if m4_wsd is not None:
            metrics_list.append(("M4_weight_sd", m4_wsd))
        for label, arr in metrics_list:
            writer.writerow([
                label,
                round(float(np.mean(arr)), 4),
                round(float(np.min(arr)), 4),
                round(float(np.max(arr)), 4),
                optode_names[int(np.argmin(arr))],
                optode_names[int(np.argmax(arr))],
            ])

        # ── Inter-optode edge distances ─────────────────────────────
        if edge_dists:
            writer.writerow([])
            writer.writerow(["# Inter-Optode Edge Distances"])
            writer.writerow([
                "optode1", "optode2", "edge_type",
                "rest_length_mm", "n_subjects",
                "mean_dist_mm", "sd_dist_mm",
                "min_dist_mm", "max_dist_mm",
                "mean_deviation_mm",
            ])
            for ed in edge_dists:
                writer.writerow([
                    ed["optode1"], ed["optode2"], ed["edge_type"],
                    round(ed["rest_length"], 4),
                    ed["n_subjects"],
                    round(ed["mean_dist"], 4),
                    round(ed["sd_dist"], 4),
                    round(ed["min_dist"], 4),
                    round(ed["max_dist"], 4),
                    round(ed["mean_deviation"], 4),
                ])

            writer.writerow([])
            writer.writerow(["# Edge Distance Summary"])
            writer.writerow(["edge_type", "n_edges",
                             "mean_abs_deviation_mm", "mean_sd_mm"])
            for typ in ("anchor", "stiff", "stiff*", "flexible"):
                subset = [e for e in edge_dists if e["edge_type"] == typ]
                if subset:
                    writer.writerow([
                        typ, len(subset),
                        round(float(np.mean([abs(e["mean_deviation"])
                                             for e in subset])), 4),
                        round(float(np.mean([e["sd_dist"]
                                             for e in subset])), 4),
                    ])

        # ── AtlasViewer-comparable section ───────────────────────────
        # Centroid-translated world coordinates, sample SD (ddof=1),
        # no anatomical normalization — matches AtlasViewer's
        # plotProbePlacementVariation.m methodology.
        if world_pos is not None:
            n_opt, _, ns = world_pos.shape

            centroids = np.mean(world_pos, axis=0)
            centered = world_pos.copy()
            for s in range(ns):
                centered[:, :, s] -= centroids[:, s]

            av_mean = np.mean(centered, axis=2)
            av_sd_per_axis = np.std(centered, axis=2, ddof=1)
            av_sd = np.zeros(n_opt)
            for i in range(n_opt):
                diffs = centered[i, :, :].T - av_mean[i]
                errs = np.sqrt(np.sum(diffs ** 2, axis=1))
                av_sd[i] = np.sqrt(np.sum(errs ** 2) / (ns - 1))

            writer.writerow([])
            writer.writerow([])
            writer.writerow(["# AtlasViewer-Comparable — Centroid-Translated World Coordinates (ddof=1)"])
            writer.writerow([
                "optode_name", "optode_type", "n_subjects",
                "mean_x_mm", "mean_y_mm", "mean_z_mm",
                "sd_mm", "sd_x_mm", "sd_y_mm", "sd_z_mm",
            ])

            for i, name in enumerate(optode_names):
                m = av_mean[i]
                sa = av_sd_per_axis[i]
                writer.writerow([
                    name,
                    optode_type_map.get(name, ""),
                    n_subj,
                    round(m[0], 4), round(m[1], 4), round(m[2], 4),
                    round(av_sd[i], 4),
                    round(sa[0], 4), round(sa[1], 4), round(sa[2], 4),
                ])

            writer.writerow([])
            writer.writerow(["# AtlasViewer-Comparable Summary"])
            writer.writerow(["coordinate_space", "centroid-translated world (mm)"])
            writer.writerow(["ddof", 1])
            writer.writerow(["normalization", "none (centroid translation only)"])
            writer.writerow([])
            writer.writerow(["metric", "mean", "min", "max",
                             "min_optode", "max_optode"])
            writer.writerow([
                "SD_mm",
                round(float(np.mean(av_sd)), 4),
                round(float(np.min(av_sd)), 4),
                round(float(np.max(av_sd)), 4),
                optode_names[int(np.argmin(av_sd))],
                optode_names[int(np.argmax(av_sd))],
            ])


# ═══════════════════════════════════════════════════════════════════════════════
# OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class NEUROCAPTAIN_OT_export_subject_json_file(bpy.types.Operator, ExportHelper):
    """Save subject variability JSON to a file"""
    bl_idname = "neurocaptain.export_subject_json_file"
    bl_label = "Save Subject JSON"
    bl_options = {'REGISTER', 'INTERNAL'}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    subject_id: StringProperty(default="sub-01")

    def execute(self, context):
        try:
            data = build_subject_data(context, self.subject_id)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)

        n_opt = len(data["optodes"])
        n_src = sum(1 for o in data["optodes"] if o["type"] == "source")
        n_det = sum(1 for o in data["optodes"] if o["type"] == "detector")
        self.report({'INFO'},
                    f"Exported {n_opt} optodes ({n_src}S/{n_det}D) "
                    f"for '{self.subject_id}' to {os.path.basename(self.filepath)}")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_export_subject_json(bpy.types.Operator):
    """Export current scene optode positions to a variability analysis JSON -
    prompts for a Subject ID, then opens a file-save dialog"""
    bl_idname = "neurocaptain.export_subject_json"
    bl_label = "Export Subject Probe Evaluation"
    bl_options = {'REGISTER'}

    subject_id: StringProperty(
        name="Subject ID",
        description="Identifier for this subject/atlas",
        default="sub-01",
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=350)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "subject_id")

    def execute(self, context):
        try:
            build_subject_data(context, self.subject_id)
        except RuntimeError as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}

        bpy.ops.neurocaptain.export_subject_json_file(
            'INVOKE_DEFAULT', subject_id=self.subject_id
        )
        return {'FINISHED'}


class NEUROCAPTAIN_OT_run_variability(bpy.types.Operator):
    """Load subject JSONs from a directory and run group variability analysis"""
    bl_idname = "neurocaptain.run_variability"
    bl_label = "Run Group Variability Analysis"
    bl_options = {'REGISTER', 'UNDO'}

    directory: StringProperty(
        name="Subject JSON Directory",
        description="Folder containing each subject's exported probe evaluation JSON",
        subtype='DIR_PATH',
    )
    show_per_subject: BoolProperty(
        name="Show Per-Subject Positions",
        description="Also visualize each subject's individual optode positions, not just the group summary",
        default=False,
    )
    csv_filename: StringProperty(
        name="CSV Filename",
        description="Filename for the summary CSV (saved in the selected directory)",
        default="group_variability_summary.csv",
    )
    probe_json_path: StringProperty(
        name="Probe JSON",
        description="Path to probe_config JSON with registration data for barycentric consistency. Auto-detected if left blank.",
        subtype='FILE_PATH',
    )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "show_per_subject")
        layout.separator()
        layout.prop(self, "csv_filename")
        layout.separator()
        layout.prop(self, "probe_json_path")

    def execute(self, context):
        if not self.directory or not os.path.isdir(self.directory):
            self.report({'ERROR'}, "Invalid directory.")
            return {'CANCELLED'}

        subjects = load_subject_directory(self.directory)
        if len(subjects) < 2:
            self.report({'ERROR'},
                        f"Need at least 2 subject JSONs, found {len(subjects)}.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Loaded {len(subjects)} subjects.")

        try:
            optode_names, nm_mm, nm_scaled, world_pos, subject_ids, scales = \
                align_subjects(subjects, exclude_short_sep=False)
        except KeyError as e:
            self.report({'ERROR'}, f"Optode alignment failed: {e}")
            return {'CANCELLED'}

        if len(optode_names) == 0:
            self.report({'ERROR'}, "No common optodes found across subjects.")
            return {'CANCELLED'}

        n_opt = len(optode_names)
        n_subj = len(subjects)

        total_possible = len(subjects[0]["optodes"])
        if n_opt < total_possible:
            dropped = total_possible - n_opt
            self.report({'WARNING'},
                        f"{dropped} optode(s) not common to all subjects (excluded).")

        mean_scales = np.mean(scales, axis=0)
        stats = compute_intersubject_stats(nm_scaled, axis_scales=mean_scales)

        mean_world = np.mean(world_pos, axis=2)

        optode_type_map = {}
        for subj in subjects:
            for o in subj["optodes"]:
                if o["name"] not in optode_type_map:
                    optode_type_map[o["name"]] = o["type"]

        m1 = compute_m1_landmark_error(subjects, optode_names)
        m2 = compute_m2_interoptode_distance(world_pos, optode_names)
        m3 = compute_m3_barycentric_stability(subjects, optode_names)

        pjp = self.probe_json_path
        if not pjp or not os.path.isfile(pjp):
            pjp = _find_probe_json(self.directory)
        m4_rms, m4_sd, m4_max, m4_wsd = compute_m4_barycentric_consistency(
            subjects, optode_names,
            probe_json_path=pjp,
        )

        edge_dists = compute_edge_distances(
            subjects, optode_names, probe_json_path=pjp,
        )

        clear_variability_objects()

        head_mesh = _find_head_mesh()
        bvh = None
        if head_mesh:
            depsgraph = context.evaluated_depsgraph_get()
            bvh = BVHTree.FromObject(head_mesh, depsgraph)

        ellipse_axes = []
        for i in range(n_opt):
            pos = mean_world[i]
            normal = [0, 0, 1]
            if bvh:
                _loc, n_vec, _fidx, _d = bvh.find_nearest(Vector(pos))
                if n_vec:
                    normal = list(n_vec)
            error_vecs = world_pos[i, :, :].T - mean_world[i]
            semi_x, semi_y = _compute_tangent_plane_ellipse(
                pos, normal, error_vecs
            )
            ellipse_axes.append((semi_x, semi_y))

        build_variability_visualization(
            context, optode_names, optode_type_map, stats, mean_world,
            self.show_per_subject, world_pos, subject_ids, world_pos
        )

        print("\n" + "=" * 70)
        print(f"  Probe Variability Analysis — {n_subj} subjects, "
              f"{n_opt} optodes")
        print(f"  Coordinates: head-frame normalized (nm_scaled)")
        print(f"  Mean scales: X={mean_scales[0]:.1f} "
              f"Y={mean_scales[1]:.1f} Z={mean_scales[2]:.1f} mm")
        print("=" * 70)
        print(f"  {'Optode':<16} {'Type':<10} {'Mean X':>8} {'Mean Y':>8} "
              f"{'Mean Z':>8} {'SD (mm)':>8}")
        print("  " + "-" * 66)
        for i, name in enumerate(optode_names):
            m = stats["mean_pos"][i]
            print(f"  {name:<16} {optode_type_map.get(name, ''):10} "
                  f"{m[0]:8.2f} {m[1]:8.2f} {m[2]:8.2f} "
                  f"{stats['sd'][i]:8.3f}")
        print("=" * 70)
        print(f"  Overall mean SD: {np.mean(stats['sd']):.3f} mm")
        print(f"  Max SD: {np.max(stats['sd']):.3f} mm "
              f"({optode_names[np.argmax(stats['sd'])]})")
        print(f"  Min SD: {np.min(stats['sd']):.3f} mm "
              f"({optode_names[np.argmin(stats['sd'])]})")
        print(f"  Mean M1 (landmark error): {np.mean(m1):.3f} mm")
        print(f"  Mean M2 (interoptode SD): {np.mean(m2):.3f} mm")
        print(f"  Mean M3 (bary stability): {np.mean(m3):.3f} mm")
        print(f"  Mean Barycentric Consistency (bary displacement):  "
              f"RMS: {np.mean(m4_rms):.3f} mm, "
              f"SD: {np.mean(m4_sd):.3f} mm, "
              f"Max: {np.mean(m4_max):.3f} mm, "
              f"Weight SD: {np.mean(m4_wsd):.4f}")
        print("=" * 70 + "\n")

        if edge_dists:
            print(f"\n  Inter-optode edge distances ({len(edge_dists)} edges):")
            for typ in ("anchor", "stiff", "stiff*", "flexible"):
                subset = [e for e in edge_dists if e["edge_type"] == typ]
                if subset:
                    devs = [abs(e["mean_deviation"]) for e in subset]
                    sds = [e["sd_dist"] for e in subset]
                    print(f"    {typ:10s}  N={len(subset):2d}  "
                          f"mean|dev|={np.mean(devs):.3f} mm  "
                          f"mean SD={np.mean(sds):.3f} mm")

        csv_path = os.path.join(self.directory, self.csv_filename)
        export_summary_csv(
            csv_path, optode_names, optode_type_map, stats, n_subj,
            ellipse_axes, m1, m2, m3, world_pos=world_pos,
            m4_rms=m4_rms, m4_sd=m4_sd, m4_max=m4_max, m4_wsd=m4_wsd,
            edge_dists=edge_dists,
        )
        print(f"  CSV saved to: {csv_path}")

        self.report({'INFO'},
                    f"Analysis complete: {n_opt} optodes, "
                    f"{n_subj} subjects, mean SD={np.mean(stats['sd']):.3f} mm  "
                    f"— CSV saved to {os.path.basename(csv_path)}")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_variability_clear(bpy.types.Operator):
    """Remove all probe variability visualization objects"""
    bl_idname = "neurocaptain.variability_clear"
    bl_label = "Clear Variability Results"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        n = clear_variability_objects()
        self.report({'INFO'}, f"Removed {n} variability objects.")
        return {'FINISHED'}


def clear_variability_objects():
    """Remove all objects in the Variability_Analysis collection."""
    coll = bpy.data.collections.get(COLLECTION_NAME)
    if coll is None:
        return 0

    count = 0
    for obj in list(coll.objects):
        mesh = obj.data if obj.type == 'MESH' else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        count += 1

    bpy.data.collections.remove(coll)

    for mat in list(bpy.data.materials):
        if mat.name.startswith("Var_") or mat.name.startswith("VarSubj_"):
            if mat.users == 0:
                bpy.data.materials.remove(mat)

    return count


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

CLASSES = [
    NEUROCAPTAIN_OT_export_subject_json_file,
    NEUROCAPTAIN_OT_export_subject_json,
    NEUROCAPTAIN_OT_run_variability,
    NEUROCAPTAIN_OT_variability_clear,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
