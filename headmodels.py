import bpy
from bpy.props import EnumProperty
import addon_utils
from bpy_extras.io_utils import ImportHelper
from bpy.types import Operator, PropertyGroup
from bpy.props import StringProperty, CollectionProperty
import os
from .utils import *
import numpy as np
import jdata as jd


def _load_mesh_direct(filepath):
    """Load a .bmsh/.jmsh/.json mesh file, bypassing jdata's patched decode pipeline."""
    import json as _json
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.bmsh':
        import bjdata as bjd
        with open(filepath, 'rb') as f:
            data = bjd.loadb(f.read())
    else:
        with open(filepath, 'r') as f:
            data = _json.load(f)
    return _decode_jdata_simple(data)


def _decode_jdata_simple(obj):
    """Minimal JData annotation decoder for mesh vertex/face arrays."""
    if isinstance(obj, dict):
        if '_ArrayType_' in obj:
            return _decode_jdata_array(obj)
        return {k: _decode_jdata_simple(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_jdata_simple(v) for v in obj]
    return obj


def _decode_jdata_array(d):
    """Decode a single JData array annotation into a numpy array."""
    import base64
    DTYPES = {
        'double': np.float64, 'single': np.float32,
        'int8': np.int8, 'uint8': np.uint8,
        'int16': np.int16, 'uint16': np.uint16,
        'int32': np.int32, 'uint32': np.uint32,
        'int64': np.int64, 'uint64': np.uint64,
    }
    dtype = DTYPES.get(d.get('_ArrayType_', 'double'), np.float64)
    shape = d.get('_ArraySize_')
    order = 'C'
    if '_ArrayOrder_' in d:
        ao = d['_ArrayOrder_'].lower()
        order = 'F' if ao in ('c', 'col', 'column') else 'C'

    if '_ArrayZipData_' in d:
        raw = d['_ArrayZipData_']
        if isinstance(raw, str):
            raw = base64.b64decode(raw)
        if isinstance(raw, np.ndarray):
            raw = raw.tobytes()
        zip_type = d.get('_ArrayZipType_', 'zlib')
        if zip_type in ('zlib', 'deflate'):
            import zlib
            raw = zlib.decompress(raw)
        elif zip_type == 'gzip':
            import gzip
            raw = gzip.decompress(raw)
        elif zip_type in ('lzma', 'lzip'):
            import lzma
            raw = lzma.decompress(raw)
        arr = np.frombuffer(raw, dtype=dtype).copy()
    elif '_ArrayData_' in d:
        arr_data = d['_ArrayData_']
        if isinstance(arr_data, (bytes, bytearray)):
            arr = np.frombuffer(arr_data, dtype=dtype).copy()
        elif isinstance(arr_data, str):
            arr = np.frombuffer(base64.b64decode(arr_data), dtype=dtype).copy()
        elif isinstance(arr_data, (list, tuple)):
            arr = np.array(arr_data, dtype=dtype).flatten()
        elif isinstance(arr_data, np.ndarray):
            arr = arr_data.astype(dtype).flatten()
        else:
            return d
    else:
        return d

    if shape and len(shape) >= 2:
        arr = arr.reshape(shape, order=order)
    return arr

if bpy.app.version < (4, 0, 0):
    addon_utils.enable("io_mesh_stl")

enum_action = [
    ("ADD_HEADMESH", "add headmesh", "Import a head surface mesh from the HeadModels folder"),
    ("ADD_BRAIN1020MESH", "add brain1020mesh", "Import a ready-made 10-20/10-10/10-5 landmark mesh from the ScalpLandmarks folder"),
]


class select_model(Operator, ImportHelper):
    bl_label = "Select a head model"
    bl_description = "Import head surface meshes or scalp landmark meshes"
    bl_idname = "braincapgen.select_model"

    action: EnumProperty(
        items=[
            ("ADD_HEADMESH", "add headmesh", "add headmesh"),
            ("ADD_BRAIN1020MESH", "add brain1020mesh", "add brain1020mesh"),
        ]
    )

    filename_ext = ".json,.jmsh,.bmsh,.stl, .off,.obj"
    filter_glob: StringProperty(
        default="*.json;*.jmsh;*.bmsh;*.stl;*.off;*.obj",
        options={"HIDDEN"},
    )
    files: CollectionProperty(type=PropertyGroup)

    @classmethod
    def func(self, context):
        bpy.ops.object.select_all(action="SELECT")

        for ob in bpy.context.selected_objects:
            if (
                ob.type == "CAMERA"
                or ob.type == "LIGHT"
                or ob.type == "EMPTY"
                or ob.type == "LAMP"
                or ob.type == "SPEAKER"
                or ob.type == "CUBE"
            ):
                ob.select_set(True)
            else:
                ob.select_set(False)
        bpy.ops.object.delete()

        for o in bpy.context.scene.objects:
            if o.name == "Cube":
                o.select_set(True)
                bpy.ops.object.delete()
            else:
                pass
        return {"FINISHED"}

    @classmethod
    def description(cls, context, properties):
        hints = {}
        for item in enum_action:
            hints[item[0]] = item[2]
        return hints[properties.action]

    def execute(self, context):
        self.func(context)
        for i in self.files:
            folder = os.path.dirname(self.filepath)
            path_to_file = os.path.join(folder, i.name)
            filename = os.path.basename(path_to_file)
            root_ext = os.path.splitext(path_to_file)
            file_ex = root_ext[1]
            obs = []
        if file_ex == ".stl":
            # iterate through the selected files

            #  full path to file
            path_to_file = os.path.join(folder, i.name)
            if bpy.app.version >= (4, 0, 0):
                bpy.ops.wm.stl_import(filepath=path_to_file)
            else:
                bpy.ops.import_mesh.stl(
                    filepath=path_to_file,
                    axis_forward="-Z",
                    axis_up="Y",
                    filter_glob="*.obj;*.stl",
                )
            # Append objects to the list
            obs.append(context.selected_objects[:])
            bpy.context.object.rotation_euler[
                0
            ] = 4.71239  ## I needed this line idk if eveyone will

            obj = bpy.context.object
            obj.name = "importedmodel"

        elif file_ex == ".obj":
            #  full path to file
            path_to_file = os.path.join(folder, i.name)
            if bpy.app.version >= (4, 0, 0):
                bpy.ops.wm.obj_import(
                    filepath=path_to_file,
                    filter_glob="*.obj;*.stl",
                )
            else:
                bpy.ops.import_scene.obj(
                    filepath=path_to_file,
                    axis_forward="-Z",
                    axis_up="Y",
                    filter_glob="*.obj;*.stl",
                )
            # append to the list
            imported_objects = context.selected_objects[:]
            if imported_objects:
                imported_objects[0].name = "importedmodel"

        else:
            try:
                surfdata = _load_mesh_direct(path_to_file)
                if "MeshVertex3" in surfdata and "MeshTri3" in surfdata:
                    AddMeshFromNodeFace(
                        surfdata["MeshVertex3"],
                        (np.array(surfdata["MeshTri3"]) - 1).astype(np.int32).tolist(),
                        "importedmodel",
                    )
                elif "node" in surfdata and "face" in surfdata:
                    AddMeshFromNodeFace(
                        surfdata["node"],
                        (np.array(surfdata["face"]) - 1).astype(np.int32).tolist(),
                        "importedmodel",
                    )
                else:
                    ShowMessageBox(f"Unsupported mesh format in file. Available keys: {surfdata.keys()}",
                                   "Load Error", "ERROR")
                    return {'CANCELLED'}

                if "landmark_labels" in surfdata:
                    bpy.data.objects["importedmodel"]["_pending_landmark_labels"] = list(surfdata["landmark_labels"])

            except Exception as e:
                print(f"Error loading mesh: {e}")
                ShowMessageBox(f"Failed to load mesh file: {str(e)}",
                               "Load Error", "ERROR")
                return {'CANCELLED'}

        if self.action == "ADD_HEADMESH":
            self.add_headmesh(context=context)

        elif self.action == "ADD_BRAIN1020MESH":
            self.add_brain1020mesh(context=context)

        return {"FINISHED"}

    def invoke(self, context, event):
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        folder = "HeadModels" if self.action == "ADD_HEADMESH" else "ScalpLandmarks"
        self.filepath = os.path.join(addon_dir, folder) + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    @staticmethod
    def add_headmesh(context):
        obj = bpy.context.object

        head = bpy.data.objects["importedmodel"]
        bpy.ops.object.select_all(action="DESELECT")
        head.select_set(True)
        recenter_on_vertex_mean(head)
        head.location = (0.0, 0.0, 0.0)

        head.name = "headmesh"
        head.select_set(True)

        # Duplicate + offset without bpy.ops.object.duplicate_move(): its
        # TRANSFORM_OT_translate stage activates a modal gizmo-drawing
        # callback (ED_region_draw_cb_activate) that requires a real
        # GPU-initialized viewport region, which crashes with a native
        # EXCEPTION_ACCESS_VIOLATION/SIGSEGV under `blender --background`
        # (confirmed on both 3.4 and 4.2 - not a Python exception, so
        # temp_override can't help). Doing the duplicate and translate
        # directly via bpy.data has the same effect and has no viewport
        # dependency at all, so it works in the interactive UI, headless
        # scripting, and CI alike.
        dup = head.copy()
        dup.data = head.data.copy()
        for collection in head.users_collection:
            collection.objects.link(dup)
        offset = (0.212906, 0.0140968, 0.0237914)
        dup.location = tuple(loc + off for loc, off in zip(dup.location, offset))

        ob = bpy.context.scene.objects["headmesh.001"]
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = ob  # Make the cube the active object
        ob.select_set(True)
        ob.hide_set(True)

        return {"FINISHED"}

    @staticmethod
    def add_brain1020mesh(context):
        from .landmark_labels import (
            LANDMARK_LABELS_1020,
            LANDMARK_LABELS_1010,
            LANDMARK_LABELS_105,
        )

        brain = bpy.data.objects["importedmodel"]
        bpy.ops.object.select_all(action="DESELECT")
        brain.select_set(True)
        brain.name = "LandmarkMesh"

        # AddMeshFromNodeFace() placed this at the 3D cursor's location.
        # headmesh always snaps to world (0,0,0) regardless of the cursor
        # (see add_headmesh) - LandmarkMesh needs the same fixed target, or
        # it silently drifts away from headmesh by however far the cursor
        # happens to be from the origin at import time.
        brain.location = (0.0, 0.0, 0.0)

        # LandmarkMesh's faces exist only so optode_connect.py's barycentric
        # registration (landmark_mesh.data.polygons + BVHTree.FromObject) has
        # a real surface to interpolate across - they're not meant to be
        # looked at. Since the landmark points don't sit exactly on
        # headmesh's surface (a few mm off in normal operation), rendering
        # those faces solid z-fights against headmesh in the viewport.
        # display_type='WIRE' (same pattern optode_connect.py already uses
        # for connection objects) keeps the mesh data fully intact for the
        # BVH/barycentric lookups while never solid-rendering it.
        brain.display_type = 'WIRE'

        num_verts = len(brain.data.vertices)

        pending = brain.get("_pending_landmark_labels")
        if pending is not None:
            labels = list(pending)
            del brain["_pending_landmark_labels"]
            if len(labels) == num_verts:
                brain["landmark_labels"] = labels
                print(f"LandmarkMesh import: loaded {len(labels)} labels from file")
            else:
                print(f"LandmarkMesh import: file label count ({len(labels)}) "
                      f"!= vertex count ({num_verts}), labels not stored")
            return {"FINISHED"}

        label_map = {
            len(LANDMARK_LABELS_1020): (LANDMARK_LABELS_1020, "10-20"),
            67: (LANDMARK_LABELS_1010[:67], "10-10 (no baseplane)"),
            len(LANDMARK_LABELS_1010): (LANDMARK_LABELS_1010, "10-10"),
            len(LANDMARK_LABELS_105): (LANDMARK_LABELS_105, "10-5"),
        }

        if num_verts in label_map:
            labels, system_name = label_map[num_verts]
            brain["landmark_labels"] = list(labels)
            print(f"LandmarkMesh import: stored {len(labels)} labels ({system_name} system)")
        else:
            print(f"LandmarkMesh import: {num_verts} vertices — no matching "
                  f"landmark system found, labels not stored")
            ShowMessageBox(
                f"Imported mesh has {num_verts} vertices which doesn't match "
                f"a known 10-20/10-10/10-5 system. Landmark labels were not assigned. "
                f"Use jmsh format with embedded labels for reliable import.",
                "Landmark Label Warning", "ERROR")

        return {"FINISHED"}