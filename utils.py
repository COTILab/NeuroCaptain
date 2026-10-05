"""NeuroCaptain utility functions."""

import bpy
import os
import tempfile
import numpy as np
from mathutils import Vector

def ShowMessageBox(message="", title="Message Box", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def AddMeshFromNodeFace(node, face, name):
    # Create mesh and related object
    my_mesh = bpy.data.meshes.new(name)
    my_obj = bpy.data.objects.new(name, my_mesh)

    # Set object location in 3D space
    my_obj.location = bpy.context.scene.cursor.location

    # make collection
    rootcoll = bpy.context.scene.collection.children.get("Collection")

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
    used_indices = {vi for p in mesh.polygons for vi in p.vertices}
    if not used_indices:
        used_indices = range(len(mesh.vertices))
    verts = np.array([mesh.vertices[i].co for i in used_indices], dtype=np.float64)
    centroid = Vector(verts.mean(axis=0).tolist())
    for v in mesh.vertices:
        v.co -= centroid
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
    verts = np.array([v.co for v in mesh.vertices], dtype=np.float64)
    tris = np.array([lt.vertices for lt in mesh.loop_triangles], dtype=np.int64)

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
    for v in mesh.vertices:
        v.co -= centroid
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


