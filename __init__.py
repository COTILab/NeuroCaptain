import bpy
from .ui import NeuroCaptain_UI
from bpy.props import PointerProperty
from .file_import import file_import
from .brain1020mesh import brain1020mesh
from .decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model
from .geonode import geo_nodes
from .dual_mesh import dual_mesh
from .niifile import niifile
from .capgen import cap_generation
from .circumference import circumference_calc

bl_info = {
    "name": "NeuroCaptain",
    "author": "(c) 2023 Ashlyn McCann, (c) 2023 Qianqian Fang",
    "version": (1, 0),  # min plug-in version
    "blender": (2, 82, 0),  # min blender version
    "location": "Layout，UI",
    "description": "generate caps for fNIRS applications",
    "warning": "This plug-in requires the preinstallation of Iso2Mesh (http://iso2mesh.sf.net) and Brain2Mesh (http://mcx.space/brain2mesh/)",
    "doc_url": "nonexistent",
    "tracker_url": "nonexistent",
    "category": "User Interface",
}


def register():
    print("Registering NeuroCaptain")
    bpy.utils.register_class(NeuroCaptain_UI)
    bpy.utils.register_class(niifile)
    bpy.utils.register_class(file_import)
    bpy.utils.register_class(decimate_mesh)
    bpy.utils.register_class(insert_shape)
    bpy.utils.register_class(brain1020mesh)
    bpy.utils.register_class(select_model)
    bpy.utils.register_class(geo_nodes)
    bpy.utils.register_class(dual_mesh)
    bpy.utils.register_class(cap_generation)
    bpy.utils.register_class(circumference_calc)
    bpy.types.Scene.neurocaptain = PointerProperty(type=niifile)


def unregister():
    print("Unregistering NeuroCaptain")
    bpy.utils.unregister_class(NeuroCaptain_UI)
    del bpy.types.Scene.blender_photonics
