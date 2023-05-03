import bpy
from .ui import BrainCapGen_UI
from bpy.props import PointerProperty
from .niifile import niifile
from .obj2surf import object2surf
from .nii2mesh import nii2mesh
from .stlfile import stl_file_import
from .initpoints import init_points
from.decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model

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
    bpy.utils.register_class(nii2mesh)
    bpy.utils.register_class(stl_file_import)
    bpy.utils.register_class(decimate_mesh)
    bpy.utils.register_class(insert_shape)
    bpy.utils.register_class(init_points)
    bpy.utils.register_class(select_model)


    bpy.types.Scene.braincapgen = PointerProperty(type=niifile)


def unregister():
    print("Unregistering BrainCapGen")
    bpy.utils.unregister_class(BrainCapGen_UI)
    del bpy.types.Scene.blender_photonics
