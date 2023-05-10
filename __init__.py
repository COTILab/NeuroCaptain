import bpy
from .ui import BrainCapGen_UI
from bpy.props import PointerProperty
from .obj2surf import object2surf
from .file_import import file_import
from .brain1020mesh import brain1020mesh
from.decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model
from .geonode import geo_nodes
from .dual_mesh import dual_mesh
from .niifile import niifile


bl_info = {
    "name": "BrainCapGen",
    "author": "(c) 2023 Ashlyn McCann, (c) 2023 Qianqian Fang",
    "version": (1, 0),  # min plug-in version
    "blender": (2, 82, 0),  # min blender version
    "location": "Layout，UI",
    "description": "generate caps for fNIRS applications",
    "warning": "This plug-in requires the preinstallation of Iso2Mesh (http://iso2mesh.sf.net) and Brain2Mesh (http://mcx.space/brain2mesh/)",
    "doc_url": "nonexistent", # add this in someday
    "tracker_url": "nonexistent", #add this someday
    "category": "User Interface",
}

def register():
    print("Registering BrainCapGen")
    bpy.utils.register_class(BrainCapGen_UI)
    #bpy.utils.register_class(object2surf)    
    bpy.utils.register_class(niifile)
    #bpy.utils.register_class(nii2mesh)
    bpy.utils.register_class(file_import)
    bpy.utils.register_class(decimate_mesh)
    bpy.utils.register_class(insert_shape)
    bpy.utils.register_class(brain1020mesh)
    bpy.utils.register_class(select_model)
    bpy.utils.register_class(geo_nodes)
    bpy.utils.register_class(dual_mesh)
    bpy.types.Scene.braincapgen = PointerProperty(type=niifile)


def unregister():
    print("Unregistering BrainCapGen")
    bpy.utils.unregister_class(BrainCapGen_UI)
    del bpy.types.Scene.blender_photonics
