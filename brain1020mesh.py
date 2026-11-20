import bpy
from bpy import context
from .utils import *
from .dependencies import safe_import, require_dependency, show_error_message
import numpy as np
import jdata as jd
import pathlib
from bpy.types import Operator, PropertyGroup
import os
from bpy.props import EnumProperty, StringProperty, CollectionProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper


# =============================================================================
# "Ignore ears" for 10-20 generation
# =============================================================================
# brain1020 measures each arc along a planar cut of the scalp. Where a cut
# plane crosses the spot where a pinna joins the head, the cut curve detours
# around the ear and the arc gets much longer on that side (sub-41: right
# CPz->TP8 arc 235 mm vs 173 mm left; CP2/CP4/CP6 pushed down by up to 50 mm).
# Fix: run brain1020 on a copy with both ears smoothed flat, then snap the
# landmarks back onto the real head surface. Sizes are relative to the
# LPA-RPA distance so they work in any units; the iteration count is scaled
# by edge length so coarse and fine meshes get the same physical smoothing.
# Tuned on BrainWeb SIAM sub-04/05/06/18/20/38/41 (~162 mm LPA-RPA, 2.3 mm
# edges -> ~65 mm zone, 10 mm fade, ~240 passes): fixes the ear detours in
# sub-41 and sub-20 (CP within 5-9 mm of the C-P midpoint on both sides)
# without changing subjects that were already fine.
EAR_ZONE_FRAC = 0.40       # zone radius around LPA and RPA
EAR_FADE_FRAC = 0.062      # smoothing fades out over this width
EAR_DIFFUSE_FRAC = 0.22    # how far the smoothing spreads per point


def smooth_ears(verts, faces0, lpa, rpa):
    """Copy of verts with the regions around LPA and RPA smoothed
    (umbrella Laplacian, factor 0.5). faces0: 0-based triangles."""
    from scipy.sparse import coo_matrix, diags
    verts = np.asarray(verts, dtype=np.float64)
    faces0 = np.asarray(faces0, dtype=np.int64)
    head = float(np.linalg.norm(np.asarray(lpa) - np.asarray(rpa)))
    if head <= 0:
        return verts.copy()
    e = np.concatenate([faces0[:, [0, 1]], faces0[:, [1, 2]], faces0[:, [2, 0]]])
    edge_len = float(np.median(np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)))
    e = np.vstack([e, e[:, ::-1]])
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(verts),) * 2).tocsr()
    adj.data[:] = 1.0
    avg = diags(1.0 / np.maximum(adj.sum(axis=1).A1, 1)) @ adj

    d_ear = np.minimum(np.linalg.norm(verts - lpa, axis=1),
                       np.linalg.norm(verts - rpa, axis=1))
    zone, fade = EAR_ZONE_FRAC * head, EAR_FADE_FRAC * head
    w = np.clip((zone - d_ear) / fade, 0.0, 1.0)[:, None]
    iters = int(np.clip((EAR_DIFFUSE_FRAC * head / max(edge_len, 1e-9)) ** 2, 20, 2000))
    out = verts.copy()
    for _ in range(iters):
        out += 0.5 * w * (avg @ out - out)
    print(f"brain1020_mesh: ignore ears - smoothed {int((w > 0).sum())} vertices "
          f"({iters} passes, max move {np.linalg.norm(out - verts, axis=1).max():.1f})")
    return out


def project_onto_surface(points, verts, faces0):
    """Closest point on the triangle surface for each point."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    bvh = BVHTree.FromPolygons([Vector(p) for p in np.asarray(verts, dtype=np.float64)],
                               np.asarray(faces0).tolist())
    out = []
    for p in np.atleast_2d(points):
        loc = bvh.find_nearest(Vector(p))[0]
        out.append(tuple(loc) if loc is not None else tuple(p))
    return np.array(out, dtype=np.float64)


enum_action = [
    ("NZ_SELECT", "nz_select", "Assign the currently-selected headmesh vertex as the Nz (nasion) landmark"),
    ("LPA_SELECT", "lpa_select", "Assign the currently-selected headmesh vertex as the Lpa (left preauricular) landmark"),
    ("RPA_SELECT", "RPA_select", "Assign the currently-selected headmesh vertex as the Rpa (right preauricular) landmark"),
    ("IZ_SELECT", "iz_select", "Assign the currently-selected headmesh vertex as the Iz (inion) landmark"),
    ("CZ_SELECT", "cz_select", "Assign the currently-selected headmesh vertex as the Cz (vertex/crown) landmark"),
    (
        "BRAIN1020_MESH",
        "brain1020_mesh",
        "Set the arc step percentages, then press OK to generate the 10-20 landmark mesh",
    ),
]


class brain1020mesh(Operator):
    bl_label = "Generate 10-20 Landmarks"
    bl_description = "Assign Nz/Lpa/Rpa/Iz/Cz landmarks and generate the 10-20 landmark mesh"
    bl_idname = "braincapgen.brain1020mesh"
    action: EnumProperty(
        items=[
            ("NZ_SELECT", "nz_select", "nz_select"),
            ("LPA_SELECT", "lpa_select", "lpa_select"),
            ("RPA_SELECT", "RPA_select", "rpa_select"),
            ("IZ_SELECT", "iz_select", "iz_select"),
            ("CZ_SELECT", "cz_select", "cz_select"),
            ("BRAIN1020_MESH", "brain1020_mesh", "brain1020_mesh"),
        ]
    )
    point1: bpy.props.FloatProperty(
        name="Arc Step 1 (%)",
        description="Percentage step along the first arc - 10 gives the 10-20 system, "
        "smaller values (e.g. 5) give denser 10-10/10-5 systems",
        default=10,
    )
    point2: bpy.props.FloatProperty(
        name="Arc Step 2 (%)",
        description="Percentage step along the second arc - 10 gives the 10-20 system, "
        "smaller values (e.g. 5) give denser 10-10/10-5 systems",
        default=10,
    )

    @classmethod
    def description(cls, context, properties):
        hints = {}
        for item in enum_action:
            hints[item[0]] = item[2]
        return hints[properties.action]

    def execute(self, context):
        outputdir = GetBPWorkFolder()
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")

        if self.action == "NZ_SELECT":
            self.nz_select(context=context)

        elif self.action == "LPA_SELECT":
            self.lpa_select(context=context)

        elif self.action == "RPA_SELECT":
            self.rpa_select(context=context)

        elif self.action == "IZ_SELECT":
            self.iz_select(context=context)

        elif self.action == "CZ_SELECT":
            self.cz_select(context=context)

        elif self.action == "BRAIN1020_MESH":
            global p1, p2
            p1 = self.point1
            p2 = self.point2
            self.brain1020_mesh(
                context=context,
                save_file=context.scene.save_landmark_file,
                save_filepath=context.scene.save_landmark_filepath,
            )

        context.scene.neurocaptain_selected_action = self.action
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "point1")
        layout.prop(self, "point2")

    def invoke(self, context, event):
        if self.action == "BRAIN1020_MESH":
            return context.window_manager.invoke_props_dialog(self)
        else:
            return self.execute(context)

    @staticmethod
    def nz_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        bpy.ops.object.mode_set(mode="OBJECT")
        selectedverts_nz = [v for v in bpy.context.active_object.data.vertices if v.select]
        if len(selectedverts_nz) == 0:
            ShowMessageBox(
                "Select the vertice that corresponds to NZ and select OK", "Error", "ERROR"
            )
            context.scene.nz_assigned = False

            return {"CANCELLED"}

        elif len(selectedverts_nz) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            context.scene.nz_assigned = False
            return {"CANCELLED"}

        else:
            vert_nz = selectedverts_nz[0].co
            v_global_nz = obj.matrix_world @ vert_nz
            global nz
            nz = np.array([v_global_nz])
            context.scene.nz_assigned = True
            context.scene["saved_nz"] = nz
            return nz

    @staticmethod
    def lpa_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_lpa = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_lpa) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to LPA and select OK", "Error", "ERROR"
            )
            context.scene.lpa_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_lpa) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            context.scene.lpa_assigned = False
            return {"CANCELLED"}

        else:
            vert_lpa = selectedverts_lpa[0].co
            v_global_lpa = obj.matrix_world @ vert_lpa
            global lpa
            lpa = np.array([v_global_lpa])
            context.scene.lpa_assigned = True
            return lpa

    @staticmethod
    def rpa_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_rpa = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_rpa) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to RPA and select OK", "Error", "ERROR"
            )
            context.scene.rpa_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_rpa) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            context.scene.rpa_assigned = False
            return {"CANCELLED"}

        else:
            vert_rpa = selectedverts_rpa[0].co
            v_global_rpa = obj.matrix_world @ vert_rpa
            global rpa
            rpa = np.array([v_global_rpa])
            context.scene.rpa_assigned = True
            return rpa

    @staticmethod
    def iz_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_iz = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_iz) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to IZ and select OK", "Error", "ERROR"
            )
            context.scene.iz_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_iz) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            context.scene.iz_assigned = False
            return {"CANCELLED"}

        else:
            vert_iz = selectedverts_iz[0].co
            v_global_iz = obj.matrix_world @ vert_iz
            global iz
            iz = np.array([v_global_iz])
            context.scene.iz_assigned = True
            return iz

    @staticmethod
    def cz_select(context):
        bpy.ops.object.mode_set(mode="OBJECT")
        obj = bpy.context.view_layer.objects.active
        selectedverts_cz = [v for v in bpy.context.active_object.data.vertices if v.select]

        if len(selectedverts_cz) == 0:
            ShowMessageBox(
                "Select the vertex that corresponds to CZ and select OK", "Error", "ERROR"
            )
            context.scene.cz_assigned = False
            return {"CANCELLED"}

        elif len(selectedverts_cz) > 1:
            ShowMessageBox("Please select only 1 vertex", "Error", "ERROR")
            context.scene.cz_assigned = False
            return {"CANCELLED"}

        else:
            vert_cz = selectedverts_cz[0].co
            v_global_cz = obj.matrix_world @ vert_cz
            global cz
            cz = np.array([v_global_cz])
            context.scene.cz_assigned = True
            return cz

    @staticmethod
    def brain1020_mesh(context, save_file=False, save_filepath=""):
        pmesh = safe_import('iso2mesh')
        if pmesh is None:
            show_error_message("iso2mesh is required for this feature")
            return {'CANCELLED'}

        scipy_spatial = safe_import('scipy.spatial')
        if scipy_spatial is None:
            show_error_message("scipy is required for this feature")
            return {'CANCELLED'}
        
        outputdir = GetBPWorkFolder()
        if not os.path.isdir(outputdir):
            os.makedirs(outputdir)
        
        obj = bpy.context.view_layer.objects.active
        
        vs = np.vstack((nz, iz, lpa, rpa, cz))
        
        verts = []
        for n in range(len(obj.data.vertices)):
            vert = obj.data.vertices[n].co
            v_global = obj.matrix_world @ vert
            verts.append(v_global)
        
        faces = [(np.array(face.vertices[:]) + 1).tolist() for face in obj.data.polygons]
        
        v = np.array(verts)
        f = np.array(faces)
        
        meshdata = {
            "_DataInfo_": {
                "JMeshVersion": "0.5",
                "Comment": "Created by BlenderPhotonics (http://mcx.space/BlenderPhotonics)",
            },
            "MeshVertex3": v,
            "MeshTri3": f,
            "param": {"initpoints": vs, "p1": p1, "p2": p2},
        }
        
        jd.save(meshdata, os.path.join(outputdir, "brain1020input.jmsh"))
        
        try:
            ignore_ears = getattr(context.scene, "neurocaptain_1020_ignore_ears", True)
            slice_verts, slice_init = meshdata["MeshVertex3"], meshdata["param"]["initpoints"]
            if ignore_ears:
                slice_verts = smooth_ears(v, f - 1, vs[2], vs[3])
                slice_init = project_onto_surface(vs, slice_verts, f - 1)

            landmarks = pmesh.brain1020(
                slice_verts,
                meshdata["MeshTri3"],
                slice_init,
                meshdata["param"]["p1"],
                meshdata["param"]["p2"],
                cztol=1e-8,
                display=0
            )

            # points1 rows are in the same order as landmarks[0].keys()
            # preserve this ordering explicitly
            label_list = list(landmarks[0].keys())
            points1 = np.vstack([landmarks[0][k] for k in label_list])
            if ignore_ears:
                # back onto the real head surface
                points1 = project_onto_surface(points1, v, f - 1)

            # Expand group-level keys to per-vertex standard names
            from .landmark_labels import GROUP_TO_STANDARD
            per_vertex_labels = []
            for k in label_list:
                group_data = np.atleast_2d(np.array(landmarks[0][k]))
                n_rows = group_data.shape[0]
                standard_names = GROUP_TO_STANDARD.get(k, {}).get(n_rows)
                if standard_names:
                    per_vertex_labels.extend(standard_names)
                elif n_rows == 1:
                    per_vertex_labels.append(k.capitalize())
                else:
                    for row_idx in range(n_rows):
                        per_vertex_labels.append(f"{k}_{row_idx}")
            print(f"brain1020_mesh: {len(per_vertex_labels)} per-vertex labels")

            keep_mask = [bool(label) for label in per_vertex_labels]
            if not all(keep_mask):
                points1 = points1[keep_mask]
                per_vertex_labels = [l for l, k in zip(per_vertex_labels, keep_mask) if k]

            tet = scipy_spatial.Delaunay(points1).simplices + 1
            face = np.array(pmesh.volface(tet)[0])

            outputmesh = {
                "_DataInfo_": {
                    "JMeshVersion": "0.5",
                    "Comment": "Created by NeuroCaptain",
                },
                "MeshVertex3": points1,
                "MeshTri3": face,
                "landmark_labels": per_vertex_labels,
            }

            jd.save(outputmesh, os.path.join(outputdir, "brain1020output.jmsh"))
            
        except Exception as e:
            print(f"Error in brain1020_mesh: {e}")
            import traceback
            traceback.print_exc()
            show_error_message(f"Error generating 10-20 mesh: {str(e)}")
            return {'CANCELLED'}
        
        # Remove stale LandmarkMesh before creating new one
        # This prevents stale landmark_labels from persisting
        if "LandmarkMesh" in bpy.data.objects:
            old_mesh = bpy.data.objects["LandmarkMesh"]
            old_mesh_data = old_mesh.data
            bpy.data.objects.remove(old_mesh, do_unlink=True)
            bpy.data.meshes.remove(old_mesh_data)
        
        AddMeshFromNodeFace(
            outputmesh["MeshVertex3"],
            (np.array(outputmesh["MeshTri3"]) - 1).tolist(),
            "LandmarkMesh",
        )
        
        mesh_obj = bpy.data.objects["LandmarkMesh"]
        num_verts = len(mesh_obj.data.vertices)
        num_labels = len(per_vertex_labels)

        print(f"brain1020_mesh: {num_labels} per-vertex labels, {num_verts} vertices")

        if num_labels == num_verts:
            mesh_obj["landmark_labels"] = per_vertex_labels
        else:
            if "landmark_labels" in mesh_obj:
                del mesh_obj["landmark_labels"]
            print(f"WARNING: per-vertex label count ({num_labels}) != vertex count ({num_verts})")
            show_error_message(
                f"Landmark label mismatch: {num_labels} labels vs {num_verts} vertices. "
                f"Labels not stored — import/export will use positional fallback."
            )
        
        if save_file and save_filepath:
            save_path = bpy.path.abspath(save_filepath)
            if not save_path.endswith((".jmsh", ".json")):
                save_path += ".jmsh"
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            landmark_data = {
                "_DataInfo_": {
                    "JMeshVersion": "0.5",
                    "Comment": "Created by NeuroCaptain",
                },
                "MeshVertex3": outputmesh["MeshVertex3"],
                "MeshTri3": outputmesh["MeshTri3"],
                "landmark_labels": per_vertex_labels,
            }
            jd.save(landmark_data, save_path)
            print(f"Saved landmark file to: {save_path}")
            ShowMessageBox(f"Landmark file saved to: {save_path}", "NeuroCaptain")

        bpy.context.view_layer.objects.active = mesh_obj
        ShowMessageBox("Generating 10-20 points is completed", "BrainCapGen")


FIDUCIALS = ("nz", "iz", "lpa", "rpa", "cz")
FIDUCIAL_ALIASES = {"nz": "nz", "nasion": "nz", "iz": "iz", "inion": "iz",
                    "lpa": "lpa", "al": "lpa", "rpa": "rpa", "ar": "rpa", "cz": "cz"}
# A point further than this fraction of the head's size from its surface
# means the file and the headmesh are not in the same space/scale
MAX_SURFACE_DIST_FRACTION = 0.05


def _floats(parts):
    try:
        return np.array([float(x) for x in parts])
    except ValueError:
        return None


def read_fiducials(path):
    """
    {nz, iz, lpa, rpa, cz: xyz} from a text file with one 'label x y z' line
    per point.  Separators may be spaces, tabs, commas or a colon after the
    label; labels are case-insensitive.  Other lines are ignored.
    A file of bare 'x y z' lines is paired line by line with the labels in a
    'refpts_labels.txt' next to it (the refpts.txt / refpts_labels.txt layout).
    """
    pts, unlabelled = {}, []
    with open(path) as f:
        for line in f:
            parts = line.replace(",", " ").replace(":", " ").split()
            key = FIDUCIAL_ALIASES.get(parts[0].lower()) if parts else None
            if key and len(parts) >= 4:
                xyz = _floats(parts[1:4])
                if xyz is not None:
                    pts.setdefault(key, xyz)
            elif len(parts) == 3 and _floats(parts) is not None:
                unlabelled.append(_floats(parts))
    labels_path = os.path.join(os.path.dirname(path), "refpts_labels.txt")
    if not pts and unlabelled and os.path.isfile(labels_path):
        with open(labels_path) as f:
            labels = [l.strip() for l in f if l.strip()]
        for label, xyz in zip(labels, unlabelled):
            key = FIDUCIAL_ALIASES.get(label.lower())
            if key:
                pts.setdefault(key, xyz)
    missing = [k.upper() for k in FIDUCIALS if k not in pts]
    if missing:
        raise ValueError(f"Missing fiducial(s) in {os.path.basename(path)}: {', '.join(missing)}")
    return pts


class import_fiducials(Operator, ImportHelper):
    bl_label = "Import Fiducials"
    bl_description = ("Load Nz/Iz/LPA/RPA/Cz from a text file (one 'label x y z' line each, "
                      "in the headmesh's coordinates) and generate the 10-20 landmark mesh")
    bl_idname = "braincapgen.import_fiducials"
    filter_glob: StringProperty(default="*.txt;*.csv", options={'HIDDEN'})

    def execute(self, context):
        # Blender can't open a dialog from the file browser callback, so hand
        # the chosen file to the options popup, which generates the mesh
        bpy.ops.braincapgen.import_fiducials_options('INVOKE_DEFAULT', filepath=self.filepath)
        return {"FINISHED"}


class import_fiducials_options(Operator):
    bl_label = "Generate 10-20 Landmarks"
    bl_description = "Set the arc steps and snapping, then generate the landmark mesh from the file"
    bl_idname = "braincapgen.import_fiducials_options"
    bl_options = {'INTERNAL'}
    filepath: StringProperty(options={'HIDDEN'})
    point1: bpy.props.FloatProperty(
        name="Arc Step 1 (%)",
        description="Percentage step along the first arc - 10 gives the 10-20 system, "
        "smaller values (e.g. 5) give denser 10-10/10-5 systems",
        default=10,
    )
    point2: bpy.props.FloatProperty(
        name="Arc Step 2 (%)",
        description="Percentage step along the second arc - 10 gives the 10-20 system, "
        "smaller values (e.g. 5) give denser 10-10/10-5 systems",
        default=10,
    )
    snap_to_head: bpy.props.BoolProperty(
        name="Snap to head surface",
        description="Move each point to the closest point on the headmesh surface. Off: use the "
                    "coordinates exactly as given in the file",
        default=False,
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.label(text=os.path.basename(self.filepath))
        layout.prop(self, "point1")
        layout.prop(self, "point2")
        layout.prop(self, "snap_to_head")

    def execute(self, context):
        try:
            pts = read_fiducials(self.filepath)
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, str(e))
            return {"CANCELLED"}

        head = bpy.data.objects.get("headmesh") or context.view_layer.objects.active
        if head is None or head.type != 'MESH':
            self.report({'ERROR'}, "No head mesh: add a 'headmesh' object first")
            return {"CANCELLED"}

        from mathutils import Vector
        failure = self.check_on_head(context, head, pts)
        if failure:
            print(f"Import Fiducials: {failure}")
            self.report({'ERROR'}, failure)
            if not bpy.app.background:              # popups crash background Blender
                ShowMessageBox(failure, "Import Fiducials failed", "ERROR")
            return {"CANCELLED"}

        if self.snap_to_head:
            # Closest point on the surface triangles, not the nearest vertex:
            # vertex snapping shifted points by ~1.7 mm (median, up to ~5 mm)
            # on uneven head meshes, and every 10-20 position inherits that.
            from mathutils.bvhtree import BVHTree
            bvh = BVHTree.FromObject(head, context.evaluated_depsgraph_get())  # local space
            world, to_local = head.matrix_world, head.matrix_world.inverted()
            snapped = {}
            for k, p in pts.items():
                loc = bvh.find_nearest(to_local @ Vector(p))[0]
                snapped[k] = np.array(world @ loc) if loc is not None else np.asarray(p)
            pts = snapped

        # Same state the manual NZ/LPA/RPA/IZ/CZ buttons set
        global nz, iz, lpa, rpa, cz, p1, p2
        nz, iz, lpa, rpa, cz = (np.array([pts[k]]) for k in FIDUCIALS)
        for k in FIDUCIALS:
            setattr(context.scene, f"{k}_assigned", True)
        context.scene["saved_nz"] = nz
        p1, p2 = self.point1, self.point2

        if context.view_layer.objects.active and context.view_layer.objects.active.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.objects.active = head
        result = brain1020mesh.brain1020_mesh(
            context=context,
            save_file=context.scene.save_landmark_file,
            save_filepath=context.scene.save_landmark_filepath,
        )
        if result == {'CANCELLED'}:
            return {"CANCELLED"}
        context.scene.neurocaptain_selected_action = "BRAIN1020_MESH"
        return {"FINISHED"}

    @staticmethod
    def check_on_head(context, head, pts):
        """Error message if any point is far from the head surface, else None."""
        from mathutils import Vector
        from mathutils.bvhtree import BVHTree
        world = head.matrix_world
        corners = [world @ Vector(c) for c in head.bound_box]
        size = max(max(c[i] for c in corners) - min(c[i] for c in corners) for i in range(3))
        limit = MAX_SURFACE_DIST_FRACTION * size
        bvh = BVHTree.FromObject(head, context.evaluated_depsgraph_get())   # head's local space
        to_local = world.inverted()
        far = []
        for k in FIDUCIALS:
            p = Vector(pts[k])
            loc = bvh.find_nearest(to_local @ p)[0]
            d = (world @ loc - p).length if loc is not None else float("inf")
            if d > limit:
                far.append(f"{k.upper()} {d:.1f}")
        if not far:
            return None
        return (f"Points are too far from the '{head.name}' surface ({', '.join(far)}; "
                f"limit {limit:.1f}, 5% of the head size). The landmarks and headmesh in "
                f"Blender may not be in the same space/scale.")

def register():
    bpy.utils.register_class(brain1020mesh)
    bpy.utils.register_class(import_fiducials)
    bpy.utils.register_class(import_fiducials_options)
    bpy.types.Scene.neurocaptain_selected_action = bpy.props.StringProperty(
        name="NeuroCaptain Selected Action", default=""
    )
    bpy.types.Scene.nz_assigned = bpy.props.BoolProperty(name="NZ Assigned", default=False)
    bpy.types.Scene.lpa_assigned = bpy.props.BoolProperty(name="LPA Assigned", default=False)
    bpy.types.Scene.rpa_assigned = bpy.props.BoolProperty(name="RPA Assigned", default=False)
    bpy.types.Scene.iz_assigned = bpy.props.BoolProperty(name="IZ Assigned", default=False)
    bpy.types.Scene.cz_assigned = bpy.props.BoolProperty(name="CZ Assigned", default=False)
    bpy.types.Scene.save_landmark_file = bpy.props.BoolProperty(
        name="Save Landmark File",
        description="Also write the generated landmark mesh to disk as a .jmsh file",
        default=False,
    )
    bpy.types.Scene.save_landmark_filepath = bpy.props.StringProperty(
        name="Save Path",
        description="Where to save the landmark file",
        subtype='FILE_PATH',
        default="",
    )
    bpy.types.Scene.neurocaptain_1020_ignore_ears = bpy.props.BoolProperty(
        name="Ignore Ears",
        description=("Compute the 10-20 arcs on a copy of the head with the ears "
                     "smoothed flat, then snap the points back onto the real surface. "
                     "Stops arcs that cut through an ear from detouring around it "
                     "and shifting the landmarks on that side"),
        default=True,
    )


def unregister():
    bpy.utils.unregister_class(import_fiducials_options)
    bpy.utils.unregister_class(import_fiducials)
    bpy.utils.unregister_class(brain1020mesh)
    del bpy.types.Scene.neurocaptain_selected_action
    del bpy.types.Scene.nz_assigned
    del bpy.types.Scene.lpa_assigned
    del bpy.types.Scene.rpa_assigned
    del bpy.types.Scene.cz_assigned
    del bpy.types.Scene.iz_assigned
    del bpy.types.Scene.save_landmark_file
    del bpy.types.Scene.save_landmark_filepath
    if hasattr(bpy.types.Scene, "neurocaptain_1020_ignore_ears"):
        del bpy.types.Scene.neurocaptain_1020_ignore_ears