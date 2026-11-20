"""NeuroCaptain utility functions."""

import bpy
import os
import tempfile
import numpy as np
from mathutils import Vector


# =============================================================================
# Large-array custom properties
# =============================================================================
# Storing big numpy arrays as obj[key] = arr.tolist() turns every row into its
# own ID property (a 3.5M-tet mesh -> ~4M of them), which makes every undo
# step, autosave and .blend save copy hundreds of MB (measured: ~1.7 s per
# undo step and a 750 MB .blend for a 27 MB mesh). Instead each array is kept
# as ONE string: "<dtype>;<shape>;<base64(zlib(bytes))>" under key + '_packed'.
# Legacy list-valued keys are still read so older .blend files keep working.

PACKED_SUFFIX = "_packed"


def pack_array(arr):
    import base64, zlib
    arr = np.ascontiguousarray(arr)
    payload = base64.b64encode(zlib.compress(arr.tobytes(), 1)).decode("ascii")
    return f"{arr.dtype.str};{','.join(str(n) for n in arr.shape)};{payload}"


def unpack_array(s):
    import base64, zlib
    dt, shape, payload = s.split(";", 2)
    shape = tuple(int(n) for n in shape.split(",")) if shape else ()
    raw = zlib.decompress(base64.b64decode(payload))
    return np.frombuffer(raw, dtype=np.dtype(dt)).reshape(shape).copy()


def store_array_prop(obj, key, arr):
    """Save a numpy array on obj compactly (replaces any legacy list prop)."""
    obj[key + PACKED_SUFFIX] = pack_array(arr)
    if key in obj:
        del obj[key]


def load_array_prop(obj, key, dtype):
    """Read an array saved by store_array_prop, or a legacy list prop.
    Returns None if neither exists."""
    packed = obj.get(key + PACKED_SUFFIX)
    if packed is not None:
        return unpack_array(packed).astype(dtype, copy=False)
    if key in obj:
        return np.array(obj[key], dtype=dtype)
    return None


def has_array_prop(obj, key):
    return (key + PACKED_SUFFIX) in obj or key in obj


def fast_volface(t):
    """Drop-in replacement for iso2mesh.volface (same faces, same order,
    same 1-based owner ids). iso2mesh finds the boundary with
    np.unique(axis=0), which sorts rows as structured records - ~10 s per
    call on a 1.6M-tet head. Packing each sorted face into one int64 key
    gives the same lexicographic order with a plain integer sort (~1 s)."""
    t = np.asarray(t, dtype=np.int64)[:, :4]
    ne = len(t)
    faces = np.concatenate((t[:, [0, 1, 2]], t[:, [1, 0, 3]],
                            t[:, [0, 2, 3]], t[:, [1, 3, 2]]))
    s = np.sort(faces, axis=1)
    n = int(s.max()) + 1
    if n ** 3 >= 2 ** 63:                       # key would overflow
        import iso2mesh as i2m
        return i2m.volface(t)
    key = (s[:, 0] * n + s[:, 1]) * n + s[:, 2]
    order = np.argsort(key, kind='stable')
    k = key[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    last = np.ones(len(k), dtype=bool)
    last[:-1] = first[1:]
    single = order[first & last]                # faces owned by one tet
    return faces[single], single % ne + 1


def _vertex_coords(mesh):
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _shift_vertices(mesh, delta):
    co = _vertex_coords(mesh) - np.asarray(delta, dtype=np.float64)
    mesh.vertices.foreach_set("co", co.ravel())

def ShowMessageBox(message="", title="Message Box", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    # No window in background mode (tests/CI): a popup there crashes Blender
    if bpy.app.background or bpy.context.window is None:
        print(f"[{title}] {message}")
        return
    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def AddMeshFromNodeFace(node, face, name):
    # Create mesh and related object
    my_mesh = bpy.data.meshes.new(name)
    my_obj = bpy.data.objects.new(name, my_mesh)

    # Set object location in 3D space
    my_obj.location = bpy.context.scene.cursor.location

    # make collection (create it if the scene has none, e.g. an empty or
    # renamed startup scene, instead of failing the whole load)
    rootcoll = bpy.context.scene.collection.children.get("Collection")
    if rootcoll is None:
        rootcoll = bpy.data.collections.get("Collection") or bpy.data.collections.new("Collection")
        bpy.context.scene.collection.children.link(rootcoll)

    # Link object to the scene collection
    rootcoll.objects.link(my_obj)

    # Create object using blender function
    my_mesh.from_pydata(node, [], face)
    my_mesh.update(calc_edges=True)


def recenter_on_vertex_mean(obj):
    """Recenter obj's mesh data on the plain arithmetic mean of its vertices,
    computed directly in Python rather than
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN').

    This matches headmesh's original (pre-existing) centering convention,
    which the landmark files under ScalpLandmarks/ were generated against -
    switching headmesh to a true volume centroid instead made it diverge
    from those files' assumed reference point (worst at the crown, where
    vertex-mean and volume-centroid differ most for a head shape).

    Doing this ourselves instead of calling the operator keeps behavior
    guaranteed-identical on every Blender version: origin_set(type=
    'ORIGIN_CENTER_OF_VOLUME') was already found to compute a different
    result on Blender 4.2.22 than on 3.4/5.0 for the same watertight mesh,
    so operator-based origin computation isn't trusted here anymore even for
    a supposedly simpler type like ORIGIN_GEOMETRY.

    After this call, obj.location is the mesh's vertex mean in world space
    and the mesh's visual position/shape is unchanged. Assumes obj has
    identity rotation/scale (true right after creation, before any transform
    is applied), so local vertex coordinates equal world coordinates.

    Only vertices referenced by at least one face are averaged. Some
    callers (layered_mesh_manager.create_mesh_object for the 5-layer
    surfaces) build a mesh from a full volumetric node array while only a
    subset of those vertices are actually used by the extracted surface's
    faces - averaging every raw vertex would pull the centroid toward the
    unused interior nodes (e.g. skull/brain tissue) instead of the visible
    surface's own mean.
    """
    mesh = obj.data
    verts = _vertex_coords(mesh)
    loop_v = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_v)
    if len(loop_v):
        verts = verts[np.unique(loop_v)]
    centroid = Vector(verts.mean(axis=0).tolist())
    _shift_vertices(mesh, centroid)
    mesh.update()
    obj.location = obj.location + centroid


def _mesh_volume_centroid_local(mesh):
    """Compute a mesh's volume centroid in its own local coordinate space
    via the divergence theorem, using only vertices referenced by real
    triangles (mesh.loop_triangles) - immune to unused/orphaned vertices
    (layered_mesh_manager.create_mesh_object builds the 5-layer surfaces
    from a full volumetric node array where only a subset of vertices are
    actually used by the extracted surface's own faces)."""
    mesh.calc_loop_triangles()
    verts = _vertex_coords(mesh)
    tris = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
    mesh.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)

    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    vol6 = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    vol = vol6.sum() / 6.0

    if abs(vol) < 1e-9:
        used = np.unique(tris)
        return verts[used].mean(axis=0) if len(used) else verts.mean(axis=0)
    return (((v0 + v1 + v2) / 4.0) * vol6[:, None]).sum(axis=0) / (6.0 * vol)


def recenter_on_volume_centroid(obj):
    """Recenter obj's mesh data on its own volume centroid (see
    _mesh_volume_centroid_local), computed directly in Python rather than
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME') - that
    operator was found to give a DIFFERENT result on Blender 4.2.22 than on
    3.4/5.0 for the same watertight mesh, so this stays version-consistent.

    After this call, obj.location is the mesh's volume centroid in world
    space and the mesh's visual position/shape is unchanged. Assumes obj has
    identity rotation/scale (true right after creation, before any transform
    is applied), so local vertex coordinates equal world coordinates.

    Used for the 5-layer surfaces (Head_Surface_5L, Brain_Cortex_5L, and
    per-layer objects), not headmesh - volume centroid is robust to two
    independently-meshed files (headmesh's own .bmsh vs. an extracted
    volumetric surface) sampling the same physical surface at different
    vertex densities, unlike vertex mean (which drifts with sampling
    density and is only reliable when comparing meshes that share a
    generation pipeline, like headmesh vs. its own landmark file - see
    recenter_on_vertex_mean).
    """
    mesh = obj.data
    centroid = Vector(_mesh_volume_centroid_local(mesh).tolist())
    _shift_vertices(mesh, centroid)
    mesh.update()
    obj.location = obj.location + centroid


def compute_volume_centroid_world(obj):
    """Compute obj's mesh volume centroid in world space, read-only (does
    not move or modify obj) - used to align a separately-meshed surface
    (e.g. the 5-layer import's own extracted scalp surface) onto a
    reference object's true geometric volume centroid, without changing
    the reference object's own origin/pivot."""
    centroid_local = _mesh_volume_centroid_local(obj.data)
    return obj.matrix_world @ Vector(centroid_local.tolist())


def GetBPWorkFolder():
    if os.name == "nt":
        return os.path.join(
            tempfile.gettempdir(),
            "iso2mesh-" + os.environ.get("UserName"),
            "neurocaptain",
        )
    else:
        return os.path.join(
            tempfile.gettempdir(),
            "iso2mesh-" + os.environ.get("USER"),
            "neurocaptain",
        )


