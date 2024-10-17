"""BlenderPhotonics Utilities/Helper Functions

* Authors: (c) 2021-2022 Qianqian Fang <q.fang at neu.edu>
* License: GNU General Public License V3 or later (GPLv3)
* Website: http://mcx.space/bp

To cite this work, please use the below information

@article {BlenderPhotonics2022,
  author = {Zhang, Yuxuang and Fang, Qianqian},
  title = {{BlenderPhotonics -- a versatile environment for 3-D complex bio-tissue modeling and light transport simulations based on Blender}},
  elocation-id = {2022.01.12.476124},
  year = {2022},
  doi = {10.1101/2022.01.12.476124},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2022/01/14/2022.01.12.476124},
  eprint = {https://www.biorxiv.org/content/early/2022/01/14/2022.01.12.476124.full.pdf},
  journal = {bioRxiv}
}
"""

import bpy
import os
import tempfile
import numpy as np


def ShowMessageBox(message="", title="Message Box", icon="INFO"):
    def draw(self, context):
        self.layout.label(text=message)

    bpy.context.window_manager.popup_menu(draw, title=title, icon=icon)


def GetNodeFacefromObject(obj, istrimesh=True):
    verts = []
    for n in range(len(obj.data.vertices)):
        vert = obj.data.vertices[n].co
        v_global = obj.matrix_world @ vert
        verts.append(v_global)
    # edges = [edge.vertices[:] for edge in obj.data.edges]
    faces = [(np.array(face.vertices[:]) + 1).tolist() for face in obj.data.polygons]
    v = np.array(verts)
    print(v)
    try:
        f = np.array(faces)
        print(f)
        return {"MeshVertex3": v, "MeshTri3": f}
    except:
        f = faces
        print(f)
    return {
        "_DataInfo_": {"BlenderObjectName", obj.name},
        "MeshVertex3": v,
        "MeshTri3": f,
    }


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


def LoadReginalMesh(meshdata, name):
    n = len(meshdata.keys()) - 1

    # To import mesh.ply in batches
    bbx = {
        "min": np.array([np.inf, np.inf, np.inf]),
        "max": np.array([-np.inf, -np.inf, -np.inf]),
    }
    for i in range(0, n):
        surfkey = "MeshTri3(" + str(i + 1) + ")"
        if n == 1:
            surfkey = "MeshTri3"
        if not isinstance(meshdata[surfkey], np.ndarray):
            meshdata[surfkey] = np.asarray(meshdata[surfkey], dtype=np.uint32)
        meshdata[surfkey] -= 1
        bbx["min"] = np.amin(
            np.vstack((bbx["min"], np.amin(meshdata["MeshVertex3"], axis=0))), axis=0
        )
        bbx["max"] = np.amax(
            np.vstack((bbx["max"], np.amax(meshdata["MeshVertex3"], axis=0))), axis=0
        )
        AddMeshFromNodeFace(meshdata["MeshVertex3"], meshdata[surfkey].tolist(), name + str(i + 1))
    print(bbx)
    return bbx


def LoadTetMesh(meshdata, name):
    if not isinstance(meshdata["MeshTri3"], np.ndarray):
        meshdata["MeshTri3"] = np.asarray(meshdata["MeshTri3"], dtype=np.uint32)
    meshdata["MeshTri3"] -= 1
    AddMeshFromNodeFace(meshdata["MeshVertex3"], meshdata["MeshTri3"].tolist(), name)


def JMeshFallback(meshobj):
    if ("MeshSurf" in meshobj) and (not ("MeshTri3" in meshobj)):
        meshobj["MeshTri3"] = meshobj.pop("MeshSurf")
    if ("MeshNode" in meshobj) and (not ("MeshVertex3" in meshobj)):
        meshobj["MeshVertex3"] = meshobj.pop("MeshNode")
    return meshobj


def auto_layer_collection():
    """
    Automatically change active layer collection.
    """
    layer = bpy.context.view_layer.active_layer_collection
    layer_collection = bpy.context.view_layer.layer_collection
    if layer.hide_viewport or layer.collection.hide_viewport:
        collections = bpy.context.object.users_collection
        for c in collections:
            lc = recurLayerCollection(layer_collection, c.name)
            if not c.hide_viewport and not lc.hide_viewport:
                bpy.context.view_layer.active_layer_collection = lc


def convert_object_to_mesh(ob, apply_modifiers=True, preserve_status=True):
    try:
        ob.name
    except:
        return None
    if ob.type != "MESH":
        if not apply_modifiers:
            mod_visibility = [m.show_viewport for m in ob.modifiers]
            for m in ob.modifiers:
                m.show_viewport = False
        # ob.modifiers.update()
        # dg = bpy.context.evaluated_depsgraph_get()
        # ob_eval = ob.evaluated_get(dg)
        # me = bpy.data.meshes.new_from_object(ob_eval, preserve_all_data_layers=True, depsgraph=dg)
        me = simple_to_mesh(ob)
        new_ob = bpy.data.objects.new(ob.data.name, me)
        new_ob.location, new_ob.matrix_world = ob.location, ob.matrix_world
        if not apply_modifiers:
            for m, vis in zip(ob.modifiers, mod_visibility):
                m.show_viewport = vis
    else:
        if apply_modifiers:
            new_ob = ob.copy()
            new_me = simple_to_mesh(ob)
            new_ob.modifiers.clear()
            new_ob.data = new_me
        else:
            new_ob = ob.copy()
            new_ob.data = ob.data.copy()
            new_ob.modifiers.clear()
    bpy.context.collection.objects.link(new_ob)
    new_ob.name = "hello"
    if preserve_status:
        new_ob.select_set(False)
    else:
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        new_ob.select_set(True)
        bpy.context.view_layer.objects.active = new_ob
    return new_ob
