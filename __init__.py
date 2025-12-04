import bpy
from bpy.props import PointerProperty
from .file_import import file_import
from .brain1020mesh import brain1020mesh
from .decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model
from .geonode import geo_nodes
from .dual_mesh_nc import dual_mesh_NC
from .niifile import niifile
from .capgen import cap_generation
from .circumference import circumference_calc
from .exportmesh import exportmesh
from .customLandmarks import customLandmarks
from . import ui  # Import the module, not the classes
from . import sd_probe_import

bl_info = {
    "name": "NeuroCaptain",
    "author": "(c) 2023 Ashlyn McCann, (c) 2023 Qianqian Fang",
    "version": (1, 0),
    "blender": (2, 82, 0),
    "location": "Layout，UI",
    "description": "generate caps for fNIRS applications",
    "warning": "This plug-in requires the preinstallation of Iso2Mesh (http://iso2mesh.sf.net) and Brain2Mesh (http://mcx.space/brain2mesh/)",
    "doc_url": "nonexistent",
    "tracker_url": "nonexistent",
    "category": "User Interface",
}


def register():
    print("Registering NeuroCaptain")
    bpy.utils.register_class(niifile)
    bpy.utils.register_class(file_import)
    bpy.utils.register_class(decimate_mesh)
    bpy.utils.register_class(insert_shape)
    bpy.utils.register_class(brain1020mesh)
    bpy.utils.register_class(select_model)
    bpy.utils.register_class(geo_nodes)
    bpy.utils.register_class(dual_mesh_NC)
    bpy.utils.register_class(cap_generation)
    bpy.utils.register_class(circumference_calc)
    bpy.utils.register_class(exportmesh)
    bpy.utils.register_class(customLandmarks)
    sd_probe_import.register()  # Register SD probe operator
    ui.register()  # Register UI panels
    bpy.types.Scene.neurocaptain = PointerProperty(type=niifile)


def unregister():
    print("Unregistering NeuroCaptain")
    del bpy.types.Scene.neurocaptain
    ui.unregister()  # Unregister UI panels
    sd_probe_import.unregister()  # Unregister SD probe operator
    bpy.utils.unregister_class(customLandmarks)
    bpy.utils.unregister_class(exportmesh)
    bpy.utils.unregister_class(circumference_calc)
    bpy.utils.unregister_class(cap_generation)
    bpy.utils.unregister_class(dual_mesh_NC)
    bpy.utils.unregister_class(geo_nodes)
    bpy.utils.unregister_class(select_model)
    bpy.utils.unregister_class(brain1020mesh)
    bpy.utils.unregister_class(insert_shape)
    bpy.utils.unregister_class(decimate_mesh)
    bpy.utils.unregister_class(file_import)
    bpy.utils.unregister_class(niifile)