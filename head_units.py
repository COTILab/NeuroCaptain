"""
Head-mesh unit check.

NeuroCaptain assumes the head mesh is in millimetres: spring lengths, the
short-separation threshold, optode sizes, light-simulation optical
properties and every reported mm value depend on it.  Some meshes are not
(e.g. the NeuroJSON BrainWeb head surfaces are in 0.5 mm voxel units despite
a "mm" LengthUnit tag), which silently makes the head 2x too large.

draw_unit_warning() shows an alert in the panel when 'headmesh' has an
implausible size for millimetres, with a button that rescales it.
"""

import bpy
from mathutils import Matrix

# Largest dimension of a real head in mm, infant to large adult.
HEAD_SIZE_MM = (100.0, 300.0)
TYPICAL_HEAD_MM = 200.0
# Unit factors to try: 0.5 mm voxels, 0.1 mm, cm, m.
CANDIDATE_FACTORS = (0.5, 0.1, 10.0, 1000.0)


def head_size(obj):
    """Largest world-space dimension of an object."""
    return max(obj.dimensions) if obj is not None else 0.0


def suggested_scale(size):
    """
    Factor that converts a head of this size to mm, or None if the size is
    already plausible.  Returns 0.0 when no standard factor fits.
    """
    lo, hi = HEAD_SIZE_MM
    if size <= 0.0 or lo <= size <= hi:
        return None
    fits = [f for f in CANDIDATE_FACTORS if lo <= size * f <= hi]
    if not fits:
        return 0.0
    return min(fits, key=lambda f: abs(size * f - TYPICAL_HEAD_MM))


def head_candidates():
    """
    'headmesh' plus freshly imported meshes not yet renamed to it: objects from
    the NeuroJSON browser (tagged neurojson_dataset_key) and the file importer
    ('importedmodel').
    """
    head = bpy.data.objects.get("headmesh")
    found = [head] if head is not None else []
    found += [o for o in bpy.data.objects
              if o.type == 'MESH' and o is not head
              and ("neurojson_dataset_key" in o or o.name == "importedmodel")]
    return found


def _scale_object(obj, factor):
    """Scale an object's geometry and location about the world origin, keeping object scale 1."""
    obj.data.transform(Matrix.Scale(factor, 4))
    obj.data.update()
    obj.location = obj.location * factor


class NEUROCAPTAIN_OT_rescale_head_mesh(bpy.types.Operator):
    """Rescale the head mesh (and a LandmarkMesh in the same units) to millimetres"""
    bl_idname = "neurocaptain.rescale_head_mesh"
    bl_label = "Rescale Head to mm"
    bl_options = {'REGISTER', 'UNDO'}

    factor: bpy.props.FloatProperty(
        name="Scale Factor",
        description="Multiply head coordinates by this factor (0.5 for 0.5 mm voxel units, 10 for cm)",
        default=0.5,
        min=1e-4,
        max=1e4,
    )

    target: bpy.props.StringProperty(
        name="Object",
        description="Head object to rescale (defaults to 'headmesh')",
        default="headmesh",
    )

    def execute(self, context):
        head = bpy.data.objects.get(self.target)
        if head is None or head.type != 'MESH':
            self.report({'ERROR'}, f"No mesh object '{self.target}' found in scene")
            return {'CANCELLED'}

        scaled = []
        for obj in (head, bpy.data.objects.get("LandmarkMesh")):
            # Only rescale a LandmarkMesh that is in the same wrong units as the head
            if obj is not None and obj.type == 'MESH' and (obj is head or suggested_scale(head_size(obj)) is not None):
                _scale_object(obj, self.factor)
                scaled.append(obj.name)

        stale = [n for n in ("Optode_Connections",) if n in bpy.data.objects]
        stale += [c for c in ("Sources", "Detectors", "Landmark_Labels") if c in bpy.data.collections]
        msg = f"Scaled {', '.join(scaled)} by x{self.factor:g} -> head is {head_size(head):.0f} mm"
        if stale:
            msg += f".  Re-create {', '.join(stale)} on the rescaled head"
            self.report({'WARNING'}, msg)
        else:
            self.report({'INFO'}, msg)
        return {'FINISHED'}


def draw_unit_warning(layout):
    """Alert box + rescale button for each head mesh that does not look like it is in mm."""
    for head in head_candidates():
        size = head_size(head)
        factor = suggested_scale(size)
        if factor is None:
            continue

        box = layout.box()
        box.alert = True
        box.label(text=f"'{head.name}' is {size:.0f} units across - not in mm?", icon='ERROR')
        if factor:
            op = box.operator(NEUROCAPTAIN_OT_rescale_head_mesh.bl_idname,
                              text=f"Scale Head x{factor:g} (-> {size * factor:.0f} mm)",
                              icon='FULLSCREEN_EXIT' if factor < 1 else 'FULLSCREEN_ENTER')
            op.factor = factor
            op.target = head.name
        else:
            box.label(text=f"Expected {HEAD_SIZE_MM[0]:.0f}-{HEAD_SIZE_MM[1]:.0f} mm; rescale manually")
