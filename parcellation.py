"""
parcellation.py
Cortical parcellation import, display and region picking for NeuroCaptain.

Phase 1 of ROI coverage quantification:
  - Import a labelled cortical surface (.bmsh/.jmsh; MeshNode + MeshSurf with
    a per-face label column, as written by SIAM / jmesh_surf.py).
  - Store each face's label as an integer face attribute (LABEL_ATTR) so the
    coverage step can read it straight off the mesh.
  - One material per region (FreeSurfer DK colors), a region list in the
    Light Sim panel, and a hover/click pick tool that names the region.

Label conventions understood:
  - Desikan-Killiany compact 1-68: 1-34 left, 35-68 right, FreeSurfer aparc
    order without 'unknown' and 'corpuscallosum' (SIAM output; verified on
    sub-04 by region centroids).
  - FreeSurfer aparc codes 1001-1035 (left) / 2001-2035 (right).
  - Anything else gets generic names and generated colors.
  - Label 0 = unlabelled (grey).

The parcellation is placed in the same Blender frame as the light
simulation: if a layered mesh is loaded, the offset uses the same formula as
lightsim_neurocaptain.load_mesh_and_register_optodes
(translation = nodes.mean - ref_world_verts.mean), so the sim's
Blender<->mesh mapping applies to this object unchanged.
"""

import colorsys
import os

import bpy
import numpy as np
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Operator, PropertyGroup, UIList
from bpy_extras.io_utils import ImportHelper


PARC_OBJ_NAME = "Parcellation_DK"
BORDER_OBJ_NAME = "Parcellation_Borders"
OUTLINE_OBJ_NAME = "Parcellation_Outline"
PARC_COLLECTION = "Parcellation"
LABEL_ATTR = "parc_label"
# Region colors as a color attribute too: the sensitivity heatmap is shown in
# Solid mode with color type 'Attribute', where material colors are ignored.
COLOR_ATTR = "ParcColor"
OFFSET_MOD = "NC_Offset"
# The parcellation surface sits this far outside the heatmap cortex so the
# two coincident surfaces don't flicker.
PARC_SURFACE_OFFSET = 0.3
LINE_OFFSET = 0.15        # thin border lines above that surface
OUTLINE_WIDTH_MM = 1.5    # thick outline around the selected regions
OUTLINE_RGB = (0.0, 0.0, 0.0)
HEATMAP_OBJ_NAMES = ("Brain_Cortex_5L", "Brain_Cortex")
MAT_PREFIX = "PARC_"
GREY = (0.45, 0.45, 0.45)
UNLABELLED_RGB = (0.30, 0.30, 0.30)

# FreeSurfer aparc (Desikan-Killiany) order, index = FreeSurfer code - 1000.
FS_APARC = [
    "unknown", "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal",
    "corpuscallosum", "cuneus", "entorhinal", "fusiform", "inferiorparietal",
    "inferiortemporal", "isthmuscingulate", "lateraloccipital",
    "lateralorbitofrontal", "lingual", "medialorbitofrontal", "middletemporal",
    "parahippocampal", "paracentral", "parsopercularis", "parsorbitalis",
    "parstriangularis", "pericalcarine", "postcentral", "posteriorcingulate",
    "precentral", "precuneus", "rostralanteriorcingulate",
    "rostralmiddlefrontal", "superiorfrontal", "superiorparietal",
    "superiortemporal", "supramarginal", "frontalpole", "temporalpole",
    "transversetemporal", "insula",
]
# Compact 34-region DK order used by 1-68 labelings.
DK_COMPACT = [n for n in FS_APARC if n not in ("unknown", "corpuscallosum")]

# FreeSurferColorLUT ctx-*-<region> colors (0-255), identical for lh/rh.
FS_COLORS = {
    "unknown": (25, 5, 25), "bankssts": (25, 100, 40),
    "caudalanteriorcingulate": (125, 100, 160),
    "caudalmiddlefrontal": (100, 25, 0), "corpuscallosum": (120, 70, 50),
    "cuneus": (220, 20, 100), "entorhinal": (220, 20, 10),
    "fusiform": (180, 220, 140), "inferiorparietal": (220, 60, 220),
    "inferiortemporal": (180, 40, 120), "isthmuscingulate": (140, 20, 140),
    "lateraloccipital": (20, 30, 140), "lateralorbitofrontal": (35, 75, 50),
    "lingual": (225, 140, 140), "medialorbitofrontal": (200, 35, 75),
    "middletemporal": (160, 100, 50), "parahippocampal": (20, 220, 60),
    "paracentral": (60, 220, 60), "parsopercularis": (220, 180, 140),
    "parsorbitalis": (20, 100, 50), "parstriangularis": (220, 60, 20),
    "pericalcarine": (120, 100, 60), "postcentral": (220, 20, 20),
    "posteriorcingulate": (220, 180, 220), "precentral": (60, 20, 220),
    "precuneus": (160, 140, 180), "rostralanteriorcingulate": (80, 20, 140),
    "rostralmiddlefrontal": (75, 50, 125), "superiorfrontal": (20, 220, 160),
    "superiorparietal": (20, 180, 140), "superiortemporal": (140, 220, 220),
    "supramarginal": (80, 160, 20), "frontalpole": (100, 0, 100),
    "temporalpole": (70, 20, 170), "transversetemporal": (150, 150, 200),
    "insula": (255, 192, 32),
}


# =============================================================================
# LABEL TABLES
# =============================================================================

def _srgb_to_linear(c):
    """Blender material colors are scene-linear; LUT colors are sRGB."""
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _fs_rgb(region):
    return tuple(_srgb_to_linear(v) for v in FS_COLORS[region])


def _generic_rgb(i):
    """Well-separated colors via golden-ratio hue stepping."""
    h = (i * 0.61803398875) % 1.0
    s = 0.65 if i % 2 == 0 else 0.85
    v = 0.90 if (i // 2) % 2 == 0 else 0.70
    return colorsys.hsv_to_rgb(h, s, v)


def detect_atlas(labels):
    """Return 'DK', 'FS_APARC' or 'GENERIC' for a set of integer labels."""
    u = set(int(x) for x in np.unique(labels)) - {0}
    if u and all(1 <= x <= 68 for x in u):
        return "DK"
    if u and all((1001 <= x <= 1035) or (2001 <= x <= 2035) for x in u):
        return "FS_APARC"
    return "GENERIC"


def region_info(label, atlas, generic_index=0):
    """(name, hemisphere, linear_rgb) for one label under an atlas."""
    label = int(label)
    if label == 0:
        return "unlabelled", "", UNLABELLED_RGB
    if atlas == "DK":
        hemi = "lh" if label <= 34 else "rh"
        region = DK_COMPACT[(label - 1) % 34]
        return f"{hemi}-{region}", hemi, _fs_rgb(region)
    if atlas == "FS_APARC":
        hemi = "lh" if label < 2000 else "rh"
        region = FS_APARC[label % 1000]
        return f"{hemi}-{region}", hemi, _fs_rgb(region)
    return f"label-{label}", "", _generic_rgb(generic_index)


def build_label_table(labels):
    """Sorted list of dicts {label, name, hemi, rgb} for the labels present."""
    atlas = detect_atlas(labels)
    table = []
    for i, lab in enumerate(sorted(int(x) for x in np.unique(labels))):
        name, hemi, rgb = region_info(lab, atlas, i)
        table.append({"label": lab, "name": name, "hemi": hemi, "rgb": rgb})
    return atlas, table


# =============================================================================
# FILE LOADING
# =============================================================================

def load_parcellation_surface(filepath):
    """Read a labelled surface. Returns (nodes (N,3) float, faces (F,3)
    0-based int, labels (F,) int).

    Uses headmodels._load_mesh_direct - raw bjdata.loadb plus a minimal JData
    array decoder, bypassing jdata's patched decode pipeline - the same
    loader the head-model import already relies on.
    """
    from .headmodels import _load_mesh_direct

    data = _load_mesh_direct(filepath)
    if not isinstance(data, dict):
        raise ValueError("File did not decode to a JMesh object")

    nodes = None
    for key in ("MeshNode", "MeshVertex3"):
        if key in data:
            nodes = np.asarray(data[key], dtype=np.float64)
            break
    if nodes is None:
        raise ValueError("No MeshNode / MeshVertex3 field found")
    nodes = nodes.reshape(-1, nodes.shape[-1])[:, :3]

    surf = None
    for key in ("MeshSurf", "MeshTri3"):
        if key in data:
            surf = np.asarray(data[key])
            break
    if surf is None:
        raise ValueError("No MeshSurf / MeshTri3 field found")
    surf = surf.reshape(-1, surf.shape[-1])

    if surf.shape[1] >= 4:
        faces = surf[:, :3].astype(np.int64)
        labels = surf[:, 3].astype(np.int64)
    else:
        faces = surf.astype(np.int64)
        lab_field = next((data[k] for k in ("MeshSurfLabel", "MeshTriLabel",
                                            "MeshLabel") if k in data), None)
        if lab_field is None:
            raise ValueError("Surface has no label column (need MeshSurf v1 v2 v3 label)")
        labels = np.asarray(lab_field).astype(np.int64).ravel()
        if len(labels) != len(faces):
            raise ValueError("Label count does not match face count")

    # JMesh indices are 1-based.
    if faces.min() >= 1 and faces.max() == len(nodes):
        faces = faces - 1
    if faces.min() < 0 or faces.max() >= len(nodes):
        raise ValueError("Face indices out of range for node array")

    # Drop degenerate triangles (repeated vertex) so face order stays 1:1
    # with labels without needing mesh.validate().
    ok = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
          & (faces[:, 0] != faces[:, 2]))
    if not ok.all():
        print(f"[Parcellation] Dropped {int((~ok).sum())} degenerate faces")
        faces, labels = faces[ok], labels[ok]

    return nodes, faces, labels


def face_areas(nodes, faces):
    a, b, c = nodes[faces[:, 0]], nodes[faces[:, 1]], nodes[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


# =============================================================================
# ALIGNMENT
# =============================================================================

def _world_vertex_mean(obj):
    # matrix_world is only refreshed on a depsgraph update; an object whose
    # location was just set (e.g. by an import in the same operator) would
    # otherwise report its old position.
    bpy.context.view_layer.update()
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    m = np.array(obj.matrix_world)
    return (co @ m[:3, :3].T + m[:3, 3]).mean(axis=0)


def mesh_to_blender_offset():
    """Vector to add to raw mesh coordinates to get Blender world coordinates,
    matching the light simulation's mapping. None if no layered mesh."""
    from . import layered_mesh_manager as lmm
    if not lmm.is_mesh_loaded():
        return None, None
    ref = (bpy.data.objects.get("Head_Surface_5L")
           or bpy.data.objects.get("headmesh"))
    if ref is None:
        return None, None
    nodes = lmm.LAYERED_MESH.nodes
    # light sim: mesh = blender + (nodes.mean - ref.mean)
    return _world_vertex_mean(ref) - nodes.mean(axis=0), ref.name


# =============================================================================
# BLENDER OBJECT
# =============================================================================

def _get_or_make_material(name, rgb):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    _set_material_rgb(mat, rgb)
    return mat


def _set_material_rgb(mat, rgb, alpha=1.0):
    mat.diffuse_color = (rgb[0], rgb[1], rgb[2], alpha)
    if mat.use_nodes and mat.node_tree:
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
    _set_material_alpha(mat, alpha)


def _set_material_alpha(mat, alpha):
    """Alpha for Solid mode (diffuse_color) and EEVEE/Cycles (BSDF Alpha)."""
    c = mat.diffuse_color
    mat.diffuse_color = (c[0], c[1], c[2], alpha)
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                node.inputs["Alpha"].default_value = alpha
    try:  # removed in 4.2 (EEVEE Next blends from alpha automatically)
        mat.blend_method = 'BLEND' if alpha < 1.0 else 'OPAQUE'
        mat.show_transparent_back = False
    except (AttributeError, TypeError):
        pass


def _remove_object(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    me = obj.data if obj.type == 'MESH' else None
    bpy.data.objects.remove(obj, do_unlink=True)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def build_parcellation_object(nodes, faces, labels, table, name=PARC_OBJ_NAME,
                              location=(0.0, 0.0, 0.0)):
    """Create the labelled mesh object. Vertices stay in raw mesh coordinates;
    the alignment offset goes in obj.location."""
    _remove_object(name)

    nv, nf = len(nodes), len(faces)
    me = bpy.data.meshes.new(name)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", nodes.astype(np.float32).ravel())
    me.loops.add(3 * nf)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set("loop_start", np.arange(0, 3 * nf, 3, dtype=np.int32))
    try:  # writable in 3.x, derived from loop_start in 4.x
        me.polygons.foreach_set("loop_total", np.full(nf, 3, dtype=np.int32))
    except (AttributeError, RuntimeError, TypeError):
        pass
    me.update(calc_edges=True)
    try:
        me.polygons.foreach_set("use_smooth", np.ones(nf, dtype=bool))
    except (AttributeError, RuntimeError, TypeError):
        pass

    attr = me.attributes.new(LABEL_ATTR, 'INT', 'FACE')
    attr.data.foreach_set("value", labels.astype(np.int32))

    slot_of = {}
    for slot, row in enumerate(table):
        me.materials.append(_get_or_make_material(MAT_PREFIX + row["name"], row["rgb"]))
        slot_of[row["label"]] = slot
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    for lab, slot in slot_of.items():
        lut[lab] = slot
    me.polygons.foreach_set("material_index", lut[labels])
    col = me.color_attributes.new(COLOR_ATTR, 'FLOAT_COLOR', 'CORNER')
    try:
        me.color_attributes.active_color = col
    except (AttributeError, TypeError):
        pass
    me.update()

    obj = bpy.data.objects.new(name, me)
    coll = bpy.data.collections.get(PARC_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(PARC_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
    elif coll.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(coll)
    coll.objects.link(obj)
    obj.location = tuple(float(v) for v in location)
    return obj


def read_face_labels(obj):
    attr = obj.data.attributes.get(LABEL_ATTR)
    if attr is None:
        return None
    out = np.empty(len(obj.data.polygons), dtype=np.int32)
    attr.data.foreach_get("value", out)
    return out


# =============================================================================
# DISPLAY STATE
# =============================================================================

_SUSPEND_UPDATES = [False]


def get_parcellation_object(scene):
    ps = scene.neurocaptain_parcellation
    return bpy.data.objects.get(ps.object_name) if ps.object_name else None


def apply_region_colors(scene):
    """Push region colors / greying onto the region materials and the
    color attribute. Regions view: region colors (greyed by the Show
    filter). Overlay view: the sensitivity heatmap - all of it (Outline),
    or only on the chosen regions with the rest in faded grey (Heatmap
    Only) - or region colors if there is no heatmap yet."""
    if _SUSPEND_UPDATES[0]:
        return
    ps = scene.neurocaptain_parcellation
    mode = ps.display_mode
    any_sel = any(r.selected for r in ps.regions)
    has_active = 0 <= ps.active_index < len(ps.regions)
    shown, dimmed = {}, set()
    for i, r in enumerate(ps.regions):
        rgb = tuple(r.color)
        if ((mode == 'SELECTED' and any_sel and not r.selected)
                or (mode == 'ACTIVE' and has_active and i != ps.active_index)):
            rgb = GREY
            dimmed.add(r.material)
        shown[r.material] = rgb
        mat = bpy.data.materials.get(r.material)
        if mat is not None:
            _set_material_rgb(mat, rgb)
    obj = get_parcellation_object(scene)
    if obj is not None:
        if ps.view_mode == 'REGIONS':
            # Parcellation alone: region colors, no heatmap.
            _write_color_attribute(obj, shown, dimmed, None)
        else:
            # Color = sensitivity only; regions are shown by the border
            # ribbons. No heatmap yet -> fall back to region colors.
            # Outline: whole map in color. Fade Others: heatmap outside the
            # selected regions blended 50% toward grey. Heatmap Only: greyed
            # by the Show filter.
            heat = _heat_colors_on(obj)
            faded = set()
            if heat is not None and ps.overlay_style != 'ONLY':
                dimmed = set()
                if ps.overlay_style == 'FADE':
                    keep = outline_labels(ps)
                    if keep:
                        faded = {r.material for r in ps.regions if r.label not in keep}
            _write_color_attribute(obj, shown, dimmed, heat, faded)
    update_outline(scene)


FADE_AMOUNT = 0.5     # Fade Others: share of grey mixed into the heatmap
FADE_GREY = 0.5


def _write_color_attribute(obj, rgb_by_material, dimmed=(), heat_rgb=None,
                           faded=()):
    """Per-corner colors (slot order = material order). Without heat_rgb:
    region colors. With heat_rgb ((n_vertices, 3) heatmap colors): the
    heatmap, with regions in `dimmed` in a faded grey version of it (keeps
    the brightness pattern, no hue) and regions in `faded` blended
    FADE_AMOUNT toward grey (looks like 50% transparency over grey; the
    Solid viewport ignores real alpha in attribute colors)."""
    me = obj.data
    col = me.color_attributes.get(COLOR_ATTR)
    if col is None or len(me.polygons) == 0:
        return
    names = [m.name if m else "" for m in me.materials]
    slots = np.array([rgb_by_material.get(n, GREY) for n in names],
                     dtype=np.float32).reshape(-1, 3)
    mi = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("material_index", mi)
    tot = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", tot)
    corner_slot = np.repeat(mi, tot)
    rgb = slots[corner_slot]
    if heat_rgb is not None:
        vidx = np.empty(len(me.loops), dtype=np.int32)
        me.loops.foreach_get("vertex_index", vidx)
        is_dim = np.array([n in dimmed for n in names], dtype=bool)[corner_slot]
        rgb = heat_rgb[vidx].astype(np.float32, copy=True)
        if is_dim.any():
            lum = rgb[is_dim] @ np.array([0.299, 0.587, 0.114], np.float32)
            rgb[is_dim] = (0.25 + 0.35 * lum)[:, None]
        is_fade = np.array([n in faded for n in names], dtype=bool)[corner_slot]
        if is_fade.any():
            rgb[is_fade] = (1.0 - FADE_AMOUNT) * rgb[is_fade] + FADE_AMOUNT * FADE_GREY
    corner = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=1)
    col.data.foreach_set("color", corner.ravel())
    me.update()


_HEAT_MAP_CACHE = {}


def _heat_colors_on(parc_obj):
    """Sensitivity heatmap colors sampled at each parcellation vertex (nearest
    heatmap-surface vertex in world space), or None if no heatmap yet."""
    heat = get_heatmap_object()
    if heat is None:
        return None
    hme = heat.data
    attr = hme.color_attributes.get('Sensitivity')
    if attr is None or attr.domain != 'CORNER' or len(hme.loops) == 0:
        return None
    hcol = np.empty(len(hme.loops) * 4, dtype=np.float32)
    attr.data.foreach_get("color", hcol)
    hloop_v = np.empty(len(hme.loops), dtype=np.int32)
    hme.loops.foreach_get("vertex_index", hloop_v)

    key = (heat.name, len(hme.vertices), len(hme.loops), parc_obj.name,
           len(parc_obj.data.vertices), tuple(np.round(np.array(heat.matrix_world).ravel(), 4)),
           tuple(np.round(np.array(parc_obj.matrix_world).ravel(), 4)))
    nearest_loop = _HEAT_MAP_CACHE.get(key)
    if nearest_loop is None:
        from scipy.spatial import cKDTree
        bpy.context.view_layer.update()
        # one representative loop per heatmap vertex that is on a face
        surf_v, first_loop = np.unique(hloop_v, return_index=True)
        hco = np.empty(len(hme.vertices) * 3)
        hme.vertices.foreach_get("co", hco)
        hw = _to_world(heat, hco.reshape(-1, 3)[surf_v])
        pco = np.empty(len(parc_obj.data.vertices) * 3)
        parc_obj.data.vertices.foreach_get("co", pco)
        pw = _to_world(parc_obj, pco.reshape(-1, 3))
        _d, idx = cKDTree(hw).query(pw, workers=-1)
        nearest_loop = first_loop[idx]
        _HEAT_MAP_CACHE.clear()
        _HEAT_MAP_CACHE[key] = nearest_loop
    return hcol.reshape(-1, 4)[nearest_loop, :3]


def _to_world(obj, co):
    m = np.array(obj.matrix_world)
    return co @ m[:3, :3].T + m[:3, 3]


def _on_region_change(self, context):
    apply_region_colors(context.scene)


def populate_regions(scene, obj, table, areas_by_label, counts_by_label, atlas):
    ps = scene.neurocaptain_parcellation
    _SUSPEND_UPDATES[0] = True
    try:
        ps.regions.clear()
        for row in table:
            r = ps.regions.add()
            r.name = row["name"]
            r.label = row["label"]
            r.hemi = row["hemi"]
            r.material = MAT_PREFIX + row["name"]
            r.color = row["rgb"]
            r.selected = False
            r.n_faces = int(counts_by_label.get(row["label"], 0))
            r.area_mm2 = float(areas_by_label.get(row["label"], 0.0))
        ps.object_name = obj.name
        ps.atlas = atlas
        ps.active_index = 0
    finally:
        _SUSPEND_UPDATES[0] = False
    apply_region_colors(scene)


def import_parcellation(context, filepath, align=True):
    """Full import: load, build object, fill the region list. Returns
    (obj, message)."""
    nodes, faces, labels = load_parcellation_surface(filepath)
    offset, ref_name = (mesh_to_blender_offset() if align else (None, None))
    location = offset if offset is not None else (0.0, 0.0, 0.0)
    obj, atlas, n_regions = _register_parcellation(
        context, nodes, faces, labels, location, 'SURFACE', filepath)

    msg = f"Imported {n_regions} {atlas} regions ({len(faces):,} faces)"
    if offset is not None:
        msg += f", aligned to {ref_name}"
        warn = _alignment_check(obj)
        if warn:
            msg += f". WARNING: {warn}"
    else:
        msg += ". Not aligned: no layered mesh loaded (raw mesh coordinates)"
    return obj, msg


def _register_parcellation(context, nodes, faces, labels, location, source,
                           source_path, scale=(1.0, 1.0, 1.0)):
    """Build the display object, fill the region list, apply overlay state.
    Returns (obj, atlas, n_regions)."""
    atlas, table = build_label_table(labels)
    obj = build_parcellation_object(nodes, faces, labels, table, location=location)
    obj.scale = tuple(float(s) for s in scale)
    obj["neurocaptain_parcellation"] = atlas
    obj["parc_source"] = source_path

    areas = face_areas(nodes, faces)
    area_by = {int(l): float(a) for l, a in zip(*_group_sum(labels, areas))}
    u, c = np.unique(labels, return_counts=True)
    count_by = {int(l): int(n) for l, n in zip(u, c)}
    scene = context.scene
    scene.neurocaptain_parcellation.source = source
    populate_regions(scene, obj, table, area_by, count_by, atlas)
    _remove_object(BORDER_OBJ_NAME)
    _remove_object(OUTLINE_OBJ_NAME)
    if scene.neurocaptain_parcellation.show_borders:
        build_border_object(obj)
    apply_overlay(scene)
    return obj, atlas, sum(1 for r in table if r["label"] != 0)


def builtin_parcel_surface(nodes, elems, solver_labels, region_labels):
    """Pial-side parcel surface of a tetrahedral mesh with built-in regions.

    Takes the outer boundary of the brain (gray + white matter, solver labels
    4/5) - the same faces as Brain_Cortex_5L - and labels each face with the
    region of the tetrahedron it belongs to. Faces of non-cortical tissue
    (region 0, e.g. cerebellum) are dropped. Returns (nodes, faces 0-based,
    labels), vertices compacted, faces wound outward.
    """
    from .utils import fast_volface
    nodes = np.asarray(nodes, dtype=np.float64)[:, :3]
    elems = np.asarray(elems)
    labels = np.asarray(solver_labels).ravel()
    region = np.asarray(region_labels).ravel()

    brain_idx = np.where(np.isin(labels, (4, 5)))[0]
    if len(brain_idx) == 0:
        raise ValueError("No gray/white matter elements (solver labels 4/5)")
    tets = elems[brain_idx, :4].astype(np.int64)
    if tets.min() == 0:
        tets = tets + 1
    faces, owner = fast_volface(tets)              # both 1-based
    faces = np.asarray(faces, dtype=np.int64) - 1
    owner = brain_idx[np.asarray(owner, dtype=np.int64).ravel() - 1]

    # Closed surface: positive signed volume <=> outward winding.
    v = nodes[faces]
    vol = np.einsum('ij,ij->i', v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0
    if vol < 0:
        faces = faces[:, ::-1]

    face_labels = region[owner].astype(np.int32)
    # Where white matter reaches the brain surface (thin cortex), the face
    # belongs to a WM tet and has no region; give it its neighbours' region.
    # Faces of other unlabelled tissue (subcortical GM, cerebellum) stay 0.
    fillable = (face_labels == 0) & (labels[owner] == 5)
    nb = face_neighbors(faces)
    face_labels = fill_labels_from_neighbors(face_labels, nb, fillable)
    face_labels = despeckle_labels(face_labels, nb)
    keep = face_labels > 0
    faces, face_labels = faces[keep], face_labels[keep]
    used, inv = np.unique(faces, return_inverse=True)
    return nodes[used], inv.reshape(-1, 3), face_labels


def face_neighbors(faces):
    """(F, 3) indices of the faces sharing each face's edges (-1 = none)."""
    faces = np.asarray(faces, dtype=np.int64)
    nf = len(faces)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    fid = np.tile(np.arange(nf), 3)
    key = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind='stable')
    key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]
    a, b = fid[:-1][same], fid[1:][same]
    src, dst = np.concatenate([a, b]), np.concatenate([b, a])
    order = np.argsort(src, kind='stable')
    src, dst = src[order], dst[order]
    start = np.searchsorted(src, src)          # first index of each src run
    rank = np.arange(len(src)) - start
    nb = np.full((nf, 3), -1, dtype=np.int64)
    ok = rank < 3
    nb[src[ok], rank[ok]] = dst[ok]
    return nb


def _neighbor_labels(labels, nb):
    return np.where(nb >= 0, labels[np.maximum(nb, 0)], 0)


def fill_labels_from_neighbors(labels, nb, fillable, max_iter=200):
    """Grow labels into `fillable` unlabelled faces, majority of labelled
    neighbours first, until nothing changes."""
    labels = labels.copy()
    for _ in range(max_iter):
        todo = fillable & (labels == 0)
        if not todo.any():
            break
        L = _neighbor_labels(labels, nb)[todo]
        pick = np.where(L[:, 0] > 0, L[:, 0], np.where(L[:, 1] > 0, L[:, 1], L[:, 2]))
        pick = np.where((L[:, 1] > 0) & (L[:, 1] == L[:, 2]), L[:, 1], pick)
        if not (pick > 0).any():
            break
        idx = np.flatnonzero(todo)
        labels[idx[pick > 0]] = pick[pick > 0]
    return labels


def despeckle_labels(labels, nb, passes=2):
    """A labelled face whose neighbours agree on a different label takes it."""
    labels = labels.copy()
    for _ in range(passes):
        L = _neighbor_labels(labels, nb)
        agree = np.where((L[:, 0] > 0) & ((L[:, 0] == L[:, 1]) | (L[:, 0] == L[:, 2])), L[:, 0],
                         np.where((L[:, 1] > 0) & (L[:, 1] == L[:, 2]), L[:, 1], 0))
        change = (labels > 0) & (agree > 0) & (agree != labels)
        if not change.any():
            break
        labels[change] = agree[change]
    return labels


def _placement_like(obj, raw_nodes):
    """(location, scale) that puts raw mesh coordinates where `obj` (built
    from all of raw_nodes as vertices, e.g. Brain_Cortex_5L) displays them."""
    me = obj.data
    if len(me.vertices) != len(raw_nodes):
        return None
    s = np.array(obj.scale)
    return _world_vertex_mean(obj) - s * np.asarray(raw_nodes)[:, :3].mean(axis=0), s


def build_builtin_parcellation(context):
    """Display object for the regions built into the loaded layered mesh."""
    from . import layered_mesh_manager as lmm
    lm = lmm.LAYERED_MESH
    if not lmm.is_mesh_loaded() or lm.region_labels is None:
        raise RuntimeError("Loaded mesh has no built-in parcellation")
    nodes, faces, labels = builtin_parcel_surface(
        lm.nodes, lm.elems, lm.tissue_labels, lm.region_labels)

    cortex = next((bpy.data.objects.get(n) for n in HEATMAP_OBJ_NAMES
                   if bpy.data.objects.get(n)), None)
    placed = _placement_like(cortex, lm.nodes) if cortex is not None else None
    if placed is not None:
        location, scale = placed
    else:
        offset, _ref = mesh_to_blender_offset()
        location = offset if offset is not None else np.zeros(3)
        scale = (1.0, 1.0, 1.0)
    obj, atlas, n_regions = _register_parcellation(
        context, nodes, faces, labels, location, 'BUILTIN',
        lm.mesh_path or "", scale=scale)
    return obj, f"Built-in parcellation: {n_regions} {atlas} regions ({len(faces):,} faces)"


# =============================================================================
# OVERLAY (parcellation + sensitivity heatmap)
# =============================================================================

def get_heatmap_object():
    return next((bpy.data.objects.get(n) for n in HEATMAP_OBJ_NAMES
                 if bpy.data.objects.get(n)), None)


def apply_overlay(scene):
    """Show the parcellation over the sensitivity heatmap.

    Overlay view: the parcellation surface carries the heatmap colors
    (sampled from the cortex under each vertex) and regions are drawn as
    border ribbons, so hue always means sensitivity. The parcellation sits a
    small offset outside the heatmap surface so the two coincident surfaces
    don't flicker; outside the parcellation (e.g. cerebellum) the heatmap
    surface shows through."""
    ps = scene.neurocaptain_parcellation
    obj = get_parcellation_object(scene)
    if obj is None:
        return
    mod = obj.modifiers.get(OFFSET_MOD)
    if mod is None:
        mod = obj.modifiers.new(OFFSET_MOD, 'DISPLACE')
        mod.direction = 'NORMAL'
        mod.mid_level = 0.0
    mod.strength = PARC_SURFACE_OFFSET

    apply_region_colors(scene)

    view = ps.view_mode
    show_parc = view != 'SENSITIVITY'
    # Thin borders of every region: always on in the Overlay, optional in
    # the Regions view. Ticked regions also get the thick outline.
    want_borders = show_parc and (view == 'OVERLAY' or ps.show_borders)
    borders = bpy.data.objects.get(BORDER_OBJ_NAME)
    if borders is None and want_borders:
        borders = build_border_object(obj)
    # Children don't inherit hiding, so every object is set explicitly.
    _set_visible(obj, show_parc)
    _set_visible(borders, want_borders)
    # The cortex under the parcellation: needed for the heatmap views (it
    # also fills gaps such as the cerebellum), hidden for regions only.
    _set_visible(get_heatmap_object(), view != 'REGIONS')


def _set_visible(obj, visible):
    """Set both hide flags (global eye and outliner/view-layer eye) so the
    object's state doesn't depend on which one was clicked last."""
    if obj is None:
        return
    obj.hide_viewport = not visible
    try:
        obj.hide_set(not visible)
    except RuntimeError:  # not in the current view layer
        pass


def _on_view_change(self, context):
    """View buttons: also hide the scalp (it covers the brain) and switch
    the viewport to Solid + attribute colors, where the views are drawn."""
    for name in ('Head_Surface_5L', 'headmesh'):
        _set_visible(bpy.data.objects.get(name), False)
    screen = getattr(context, "screen", None)
    for area in (screen.areas if screen else ()):
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'VERTEX'
    apply_overlay(context.scene)


def _on_overlay_change(self, context):
    # Border ribbons follow the line offset.
    borders = bpy.data.objects.get(BORDER_OBJ_NAME)
    obj = get_parcellation_object(context.scene)
    if (borders is not None and obj is not None
            and abs(borders.get("lift", 0.0) - border_lift(self)) > 1e-6):
        build_border_object(obj)
    apply_overlay(context.scene)


def _on_borders_toggle(self, context):
    obj = get_parcellation_object(context.scene)
    if self.show_borders and obj is not None and bpy.data.objects.get(BORDER_OBJ_NAME) is None:
        build_border_object(obj)
    apply_overlay(context.scene)


def edge_label_pairs(faces, labels):
    """Unique mesh edges with the labels of the faces on either side.
    Returns (edges (E,2), label_a (E,), label_b (E,)); label_b is -1 on an
    open boundary edge (one face). Non-manifold extra faces are ignored."""
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e.sort(axis=1)
    lab = np.tile(np.asarray(labels), 3)
    key = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    order = np.argsort(key, kind='stable')
    key, e, lab = key[order], e[order], lab[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    start = np.flatnonzero(first)
    counts = np.diff(np.append(start, len(key)))
    la = lab[start]
    lb = np.where(counts > 1, lab[np.minimum(start + 1, len(lab) - 1)], -1)
    return e[start], la, lb


def region_border_edges(faces, labels):
    """Edges (vertex pairs) shared by faces of different regions."""
    edges, la, lb = edge_label_pairs(faces, labels)
    return edges[(lb >= 0) & (la != lb)]


def region_outline_edges(faces, labels, keep_labels):
    """Edges outlining each region in `keep_labels`: edges between a kept
    region and any other region, plus the kept regions' open boundary
    edges (e.g. where the cortex surface stops at the medial wall)."""
    edges, la, lb = edge_label_pairs(faces, labels)
    keep = np.asarray(sorted(keep_labels), dtype=np.int64)
    mask = (la != lb) & (np.isin(la, keep) | np.isin(lb, keep))
    return edges[mask]


BORDER_WIDTH_MM = 0.5
BORDER_RGB = (0.01, 0.01, 0.01)


def border_ribbons(co, normals, edges, width=BORDER_WIDTH_MM, lift=0.5,
                   extend=False):
    """Flat quads along each border edge, lying on the surface and lifted
    `lift` mm along the normal. Unlike wire edges, these show in renders.
    extend: lengthen each quad by half the width at both ends so thick
    ribbons overlap at corners instead of leaving notches."""
    a, b = co[edges[:, 0]], co[edges[:, 1]]
    n = normals[edges[:, 0]] + normals[edges[:, 1]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    d = b - a
    if extend:
        dh = d * ((0.5 * width) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9))
        a, b = a - dh, b + dh
    side = np.cross(n, d)
    side *= (0.5 * width) / np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    up = n * lift
    verts = np.stack([a - side + up, a + side + up, b + side + up, b - side + up], axis=1)
    quads = np.arange(4 * len(edges), dtype=np.int32).reshape(-1, 4)
    return verts.reshape(-1, 3), quads


def _mesh_tris(parc_obj):
    """(co, normals, faces) of the parcellation mesh (all triangles)."""
    me = parc_obj.data
    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    nrm = np.empty(nv * 3, dtype=np.float64)
    me.vertices.foreach_get("normal", nrm)
    vidx = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", vidx)
    return co.reshape(-1, 3), nrm.reshape(-1, 3), vidx.reshape(-1, 3)


def build_border_object(parc_obj, lift=None):
    """Dark ribbons tracing region borders, parented to the parcellation and
    lifted just above its (offset) surface."""
    if lift is None:
        lift = border_lift(bpy.context.scene.neurocaptain_parcellation)
    co, nrm, tris = _mesh_tris(parc_obj)
    edges = region_border_edges(tris, read_face_labels(parc_obj))
    verts, quads = border_ribbons(co, nrm, edges, lift=lift)
    bobj = _build_ribbon_object(parc_obj, BORDER_OBJ_NAME, verts, quads, BORDER_RGB)
    bobj["lift"] = float(lift)
    return bobj


def border_lift(ps):
    """Height of the thin border lines above the (unoffset) parcellation
    mesh: the fixed surface offset plus the user's line offset."""
    return PARC_SURFACE_OFFSET + LINE_OFFSET


def outline_active(ps):
    return ps.view_mode == 'OVERLAY' and ps.overlay_style in ('OUTLINE', 'FADE')


def outline_labels(ps):
    """Regions to outline: the highlighted row in Show=Active, otherwise the
    ticked regions."""
    if ps.display_mode == 'ACTIVE':
        if 0 <= ps.active_index < len(ps.regions):
            r = ps.regions[ps.active_index]
            return {r.label} if r.label != 0 else set()
        return set()
    return {r.label for r in ps.regions if r.selected and r.label != 0}


def update_outline(scene):
    """Thick outline ribbons around the ticked regions (Overlay view), on
    top of the thin all-region borders. Rebuilt only when regions, width,
    color or offset change; removed otherwise."""
    ps = scene.neurocaptain_parcellation
    parc = get_parcellation_object(scene)
    labels = outline_labels(ps) if (parc is not None and outline_active(ps)) else set()
    if not labels:
        _remove_object(OUTLINE_OBJ_NAME)
        return None
    lift = border_lift(ps) + 0.15           # just above the thin borders
    rgb = OUTLINE_RGB
    sig = (f"{sorted(labels)}|{OUTLINE_WIDTH_MM:.3f}|{lift:.3f}|{rgb}|"
           f"{len(parc.data.polygons)}")
    old = bpy.data.objects.get(OUTLINE_OBJ_NAME)
    if old is not None and old.get("sig") == sig and old.parent == parc:
        _set_visible(old, True)
        return old
    _remove_object(OUTLINE_OBJ_NAME)
    co, nrm, tris = _mesh_tris(parc)
    edges = region_outline_edges(tris, read_face_labels(parc), labels)
    if len(edges) == 0:
        return None
    verts, quads = border_ribbons(co, nrm, edges, width=OUTLINE_WIDTH_MM,
                                  lift=lift, extend=True)
    obj = _build_ribbon_object(parc, OUTLINE_OBJ_NAME, verts, quads, rgb)
    obj["sig"] = sig
    return obj


def _build_ribbon_object(parc_obj, name, verts, quads, rgb):
    """Mesh object from ribbon quads, colored `rgb`, parented to parc_obj."""
    _remove_object(name)
    bme = bpy.data.meshes.new(name)
    bme.vertices.add(len(verts))
    bme.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    bme.loops.add(quads.size)
    bme.loops.foreach_set("vertex_index", quads.ravel())
    bme.polygons.add(len(quads))
    bme.polygons.foreach_set("loop_start", np.arange(0, quads.size, 4, dtype=np.int32))
    try:
        bme.polygons.foreach_set("loop_total", np.full(len(quads), 4, dtype=np.int32))
    except (AttributeError, RuntimeError, TypeError):
        pass
    bme.update(calc_edges=True)
    col = bme.color_attributes.new(COLOR_ATTR, 'FLOAT_COLOR', 'CORNER')
    col.data.foreach_set("color", np.tile(np.array(tuple(rgb) + (1.0,), np.float32),
                                          quads.size))
    try:
        bme.color_attributes.active_color = col
    except (AttributeError, TypeError):
        pass
    mat_name = MAT_PREFIX + ("outline" if name == OUTLINE_OBJ_NAME else "border")
    bme.materials.append(_get_or_make_material(mat_name, rgb))

    bobj = bpy.data.objects.new(name, bme)
    for coll in parc_obj.users_collection:
        coll.objects.link(bobj)
    bobj.parent = parc_obj
    # Not show_in_front: that would also draw borders hidden in sulci and on
    # the far side. The ribbons already sit `lift` mm above the surface.
    bobj.hide_select = True
    return bobj


def _group_sum(labels, values):
    u, inv = np.unique(labels, return_inverse=True)
    return u, np.bincount(inv, weights=values)


def _alignment_check(obj, tol_mm=15.0):
    """Compare against Brain_Cortex(_5L) bounding-box centre after alignment;
    a large gap usually means the files are not in the same space."""
    cortex = (bpy.data.objects.get("Brain_Cortex_5L")
              or bpy.data.objects.get("Brain_Cortex"))
    if cortex is None:
        return None
    gap = np.linalg.norm(_world_bbox_center(obj) - _world_bbox_center(cortex))
    if gap > tol_mm:
        return (f"parcellation centre is {gap:.0f} mm from {cortex.name}; "
                "check both meshes are from the same subject/space")
    return None


def _world_bbox_center(obj):
    m = np.array(obj.matrix_world)
    bb = np.array([tuple(c) for c in obj.bound_box])
    w = bb @ m[:3, :3].T + m[:3, 3]
    return 0.5 * (w.min(axis=0) + w.max(axis=0))


# =============================================================================
# PROPERTIES
# =============================================================================

class NC_ParcRegion(PropertyGroup):
    # `name` (built in) holds the region name, e.g. "lh-precentral".
    label: IntProperty(name="Label")
    hemi: StringProperty(name="Hemisphere")
    material: StringProperty(name="Material")
    color: FloatVectorProperty(name="Color", subtype='COLOR', size=3,
                               min=0.0, max=1.0, update=_on_region_change)
    selected: BoolProperty(name="Selected",
                           description="Include this region in the user-defined ROI set",
                           default=False, update=_on_region_change)
    n_faces: IntProperty(name="Faces")
    area_mm2: FloatProperty(name="Area (mm²)")


class NC_ParcSettings(PropertyGroup):
    regions: CollectionProperty(type=NC_ParcRegion)
    active_index: IntProperty(name="Active Region", default=0,
                              update=_on_region_change)
    object_name: StringProperty(name="Parcellation Object")
    atlas: StringProperty(name="Atlas")
    display_mode: EnumProperty(
        name="Show",
        description="Which regions keep their color",
        items=[
            ('ALL', "All", "Color every region"),
            ('SELECTED', "Selected", "Only ticked regions keep their color; others are dimmed"),
            ('ACTIVE', "Active", "Only the highlighted row keeps its color (and is "
             "outlined in the Overlay); others are dimmed"),
        ],
        default='ALL', update=_on_region_change,
    )
    source: EnumProperty(
        name="Source",
        items=[('NONE', "None", ""),
               ('BUILTIN', "Built into head mesh", "Regions stored in the tetrahedral mesh (dkmerge)"),
               ('SURFACE', "Separate surface", "Imported parcellation surface file")],
        default='NONE',
    )
    # ── Overlay with the sensitivity heatmap ─────────────────────────────
    view_mode: EnumProperty(
        name="View",
        description="What to show on the brain",
        items=[
            ('OVERLAY', "Overlay", "Sensitivity heatmap with region borders; ticked regions outlined thicker"),
            ('REGIONS', "Regions", "Parcellation only (region colors)"),
            ('SENSITIVITY', "Sensitivity", "Sensitivity heatmap only"),
        ],
        default='OVERLAY', update=_on_view_change,
    )
    overlay_style: EnumProperty(
        name="Selected Regions",
        description="How the chosen regions are shown on the sensitivity map",
        items=[
            ('OUTLINE', "Outline",
             "Full heatmap everywhere; ticked regions (or the highlighted row in "
             "Show=Active) get a thick outline"),
            ('FADE', "Fade Others",
             "Selected regions in full color with a thick outline; the heatmap "
             "elsewhere is faded 50% so the ROI stands out but outside "
             "sensitivity stays visible"),
            ('ONLY', "Heatmap Only",
             "Heatmap only on the regions in the Show filter; everything else greyed"),
        ],
        default='OUTLINE', update=_on_overlay_change,
    )
    show_borders: BoolProperty(
        name="Region Borders",
        description="Outline regions in the Regions view (always on in Overlay)",
        default=False, update=_on_borders_toggle,
    )


# =============================================================================
# UI LIST
# =============================================================================

class NEUROCAPTAIN_UL_parc_regions(UIList):
    bl_idname = "NEUROCAPTAIN_UL_parc_regions"

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "selected", text="")
        sw = row.row(align=True)
        sw.scale_x = 0.35
        sw.prop(item, "color", text="")
        row.label(text=f"{item.label:>3}  {item.name}")


# =============================================================================
# OPERATORS
# =============================================================================

class NEUROCAPTAIN_OT_import_parcellation(Operator, ImportHelper):
    """Import a labelled cortical parcellation surface (.bmsh/.jmsh)"""
    bl_idname = "neurocaptain.import_parcellation"
    bl_label = "Import Parcellation"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".bmsh"
    filter_glob: StringProperty(default="*.bmsh;*.jmsh;*.json", options={'HIDDEN'})
    align_to_sim: BoolProperty(
        name="Align to light-sim mesh",
        description=("Place the parcellation in the same frame as the loaded "
                     "layered mesh, using the light simulation's mapping"),
        default=True,
    )

    def execute(self, context):
        if not os.path.isfile(self.filepath):
            self.report({'ERROR'}, f"File not found: {self.filepath}")
            return {'CANCELLED'}
        try:
            _obj, msg = import_parcellation(context, self.filepath, self.align_to_sim)
        except Exception as e:  # surfaced to the user, full trace in console
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Parcellation import failed: {e}")
            return {'CANCELLED'}
        self.report({'WARNING'} if "WARNING" in msg or "Not aligned" in msg else {'INFO'}, msg)
        return {'FINISHED'}


def _builtin_available():
    from . import layered_mesh_manager as lmm
    return lmm.is_mesh_loaded() and lmm.LAYERED_MESH.region_labels is not None


class NEUROCAPTAIN_OT_import_parcellated_mesh(Operator, ImportHelper):
    """Import a tetrahedral head model with the cortical parcellation built in
    (v2m dkmerge: tissues 1-5 plus regions as 100 + label)"""
    bl_idname = "neurocaptain.import_parcellated_mesh"
    bl_label = "Import Parcellated Head Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".bmsh"
    filter_glob: StringProperty(default="*.bmsh;*.jmsh;*.json", options={'HIDDEN'})

    def execute(self, context):
        from . import layered_mesh_manager as lmm
        nc = getattr(context.scene, "neurocaptain_settings", None)
        order = getattr(nc, "tissue_label_order", 'AUTO') if nc else 'AUTO'
        try:
            result = lmm.import_parcellated_head_model(self.filepath, 'headmesh', order)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, f"Import failed: {e}")
            return {'CANCELLED'}
        if not result.get('success'):
            self.report({'ERROR'}, result.get('message', "Import failed"))
            return {'CANCELLED'}
        try:
            _obj, msg = build_builtin_parcellation(context)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'WARNING'}, f"{result['message']}; parcellation display failed: {e}")
            return {'FINISHED'}
        self.report({'INFO'}, f"{result['message']}. {msg}")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_show_builtin_parcellation(Operator):
    """Show the regions built into the loaded head mesh"""
    bl_idname = "neurocaptain.show_builtin_parcellation"
    bl_label = "Show Built-in Parcellation"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _builtin_available()

    def execute(self, context):
        try:
            _obj, msg = build_builtin_parcellation(context)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class NEUROCAPTAIN_OT_parc_overlay_view(Operator):
    """Show the parcellation and the sensitivity heatmap together in the
    viewport (Solid shading, attribute colors)"""
    bl_idname = "neurocaptain.parc_overlay_view"
    bl_label = "Overlay View"

    @classmethod
    def poll(cls, context):
        return get_parcellation_object(context.scene) is not None

    def execute(self, context):
        ps = context.scene.neurocaptain_parcellation
        if ps.view_mode != 'OVERLAY':
            ps.view_mode = 'OVERLAY'   # update callback does the rest
        else:
            _on_view_change(ps, context)
        heat = get_heatmap_object()
        if heat is None or 'Sensitivity' not in heat.data.color_attributes:
            self.report({'INFO'}, "No sensitivity map yet - run MMC or Redbird to overlay it")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_parc_select(Operator):
    """Select or deselect regions in bulk"""
    bl_idname = "neurocaptain.parc_select"
    bl_label = "Select Regions"
    bl_options = {'REGISTER', 'UNDO'}

    action: EnumProperty(items=[
        ('ALL', "All", "Select all regions"),
        ('NONE', "None", "Deselect all regions"),
        ('LEFT', "Left", "Select only left-hemisphere regions"),
        ('RIGHT', "Right", "Select only right-hemisphere regions"),
        ('INVERT', "Invert", "Invert selection"),
    ])

    def execute(self, context):
        ps = context.scene.neurocaptain_parcellation
        _SUSPEND_UPDATES[0] = True
        try:
            for r in ps.regions:
                if self.action == 'ALL':
                    r.selected = r.label != 0
                elif self.action == 'NONE':
                    r.selected = False
                elif self.action == 'LEFT':
                    r.selected = r.hemi == "lh"
                elif self.action == 'RIGHT':
                    r.selected = r.hemi == "rh"
                elif self.action == 'INVERT':
                    r.selected = (not r.selected) and r.label != 0
        finally:
            _SUSPEND_UPDATES[0] = False
        apply_region_colors(context.scene)
        return {'FINISHED'}


class NEUROCAPTAIN_OT_parc_pick(Operator):
    """Hover over the parcellation to see region names. Click: make active.
    Shift+click: select/deselect. Right-click or Esc: exit"""
    bl_idname = "neurocaptain.parc_pick"
    bl_label = "Pick Region"

    @classmethod
    def poll(cls, context):
        return (context.area is not None and context.area.type == 'VIEW_3D'
                and get_parcellation_object(context.scene) is not None)

    def invoke(self, context, event):
        from mathutils.bvhtree import BVHTree
        obj = get_parcellation_object(context.scene)
        self._labels = read_face_labels(obj)
        if self._labels is None:
            self.report({'ERROR'}, "Parcellation object has no label attribute")
            return {'CANCELLED'}
        ps = context.scene.neurocaptain_parcellation
        if ps.view_mode == 'SENSITIVITY':
            ps.view_mode = 'OVERLAY'   # can't pick on a hidden surface
        depsgraph = context.evaluated_depsgraph_get()
        self._bvh = BVHTree.FromObject(obj, depsgraph)
        self._obj = obj
        ps = context.scene.neurocaptain_parcellation
        self._index_of = {r.label: i for i, r in enumerate(ps.regions)}
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(
            "Pick Region: hover to identify | Click: highlight | "
            "Shift+Click: select/deselect | Right-click/Esc: done")
        context.area.header_text_set("Hover over the parcellation")
        return {'RUNNING_MODAL'}

    def _region_under_mouse(self, context, event):
        from bpy_extras import view3d_utils
        region, rv3d = context.region, context.region_data
        if region is None or rv3d is None:
            return None
        coord = (event.mouse_region_x, event.mouse_region_y)
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        inv = self._obj.matrix_world.inverted()
        o_l = inv @ origin
        d_l = (inv.to_3x3() @ direction).normalized()
        _loc, _nrm, face, _dist = self._bvh.ray_cast(o_l, d_l)
        if face is None or face >= len(self._labels):
            return None
        return self._index_of.get(int(self._labels[face]))

    def _finish(self, context):
        context.area.header_text_set(None)
        context.workspace.status_text_set(None)

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            self._finish(context)
            return {'FINISHED'}
        if context.area is None:
            return {'PASS_THROUGH'}
        ps = context.scene.neurocaptain_parcellation
        if event.type == 'MOUSEMOVE':
            idx = self._region_under_mouse(context, event)
            if idx is None:
                context.area.header_text_set("No region under cursor")
            else:
                r = ps.regions[idx]
                tick = "  [selected]" if r.selected else ""
                context.area.header_text_set(
                    f"{r.name}   (label {r.label}, {r.area_mm2:.0f} mm²){tick}")
            return {'PASS_THROUGH'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            idx = self._region_under_mouse(context, event)
            if idx is None:
                return {'PASS_THROUGH'}
            r = ps.regions[idx]
            if event.shift:
                r.selected = not r.selected
            else:
                ps.active_index = idx
            self.report({'INFO'}, f"{r.name} (label {r.label})")
            for area in context.screen.areas:
                area.tag_redraw()
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}


# =============================================================================
# PANEL SECTION (drawn inside the Light Simulation panel)
# =============================================================================

def draw_parcellation_section(layout, context):
    ps = context.scene.neurocaptain_parcellation
    box = layout.box()
    box.label(text="Cortical Parcellation", icon="MOD_TRIANGULATE")
    obj = get_parcellation_object(context.scene)
    builtin = _builtin_available()
    if obj is None:
        if builtin:
            box.operator(NEUROCAPTAIN_OT_show_builtin_parcellation.bl_idname,
                         text="Show Built-in Parcellation", icon="HIDE_OFF")
        box.operator(NEUROCAPTAIN_OT_import_parcellation.bl_idname,
                     text="Import Separate Surface (.bmsh)", icon="IMPORT")
        return

    n = sum(1 for r in ps.regions if r.label != 0)
    n_sel = sum(1 for r in ps.regions if r.selected)
    src = {'BUILTIN': "built-in", 'SURFACE': "separate surface"}.get(ps.source, "")
    row = box.row()
    row.label(text=f"{n} {ps.atlas} regions ({src}), {n_sel} selected")
    box.row().prop(ps, "view_mode", expand=True)

    row = box.row(align=True)
    row.operator(NEUROCAPTAIN_OT_parc_pick.bl_idname, text="Pick Region",
                 icon="EYEDROPPER")
    if builtin and ps.source != 'BUILTIN':
        row.operator(NEUROCAPTAIN_OT_show_builtin_parcellation.bl_idname,
                     text="", icon="MESH_ICOSPHERE")
    row.operator(NEUROCAPTAIN_OT_import_parcellation.bl_idname, text="",
                 icon="FILE_REFRESH")

    heat = get_heatmap_object()
    if (ps.view_mode != 'REGIONS'
            and (heat is None or 'Sensitivity' not in heat.data.color_attributes)):
        box.label(text="No sensitivity map yet - run MMC or Redbird", icon="INFO")
    # Outline width/color and line offset stay at their defaults (not shown).
    if ps.view_mode == 'OVERLAY':
        box.row().prop(ps, "overlay_style", expand=True)
    elif ps.view_mode == 'REGIONS':
        box.row().prop(ps, "show_borders", toggle=True, icon="MOD_WIREFRAME")

    box.prop(ps, "display_mode", expand=True)
    if outline_active(ps) and not outline_labels(ps):
        box.label(text="Tick regions below to outline them", icon="INFO")
    row = box.row(align=True)
    for action, label in (('ALL', "All"), ('NONE', "None"), ('LEFT', "Left"),
                          ('RIGHT', "Right"), ('INVERT', "Invert")):
        row.operator(NEUROCAPTAIN_OT_parc_select.bl_idname, text=label).action = action
    box.template_list(NEUROCAPTAIN_UL_parc_regions.bl_idname, "", ps, "regions",
                      ps, "active_index", rows=8)


# =============================================================================
# REGISTER
# =============================================================================

CLASSES = [
    NC_ParcRegion,
    NC_ParcSettings,
    NEUROCAPTAIN_UL_parc_regions,
    NEUROCAPTAIN_OT_import_parcellation,
    NEUROCAPTAIN_OT_import_parcellated_mesh,
    NEUROCAPTAIN_OT_show_builtin_parcellation,
    NEUROCAPTAIN_OT_parc_overlay_view,
    NEUROCAPTAIN_OT_parc_select,
    NEUROCAPTAIN_OT_parc_pick,
]


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.neurocaptain_parcellation = PointerProperty(type=NC_ParcSettings)


def unregister():
    if hasattr(bpy.types.Scene, "neurocaptain_parcellation"):
        del bpy.types.Scene.neurocaptain_parcellation
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
