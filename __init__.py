import bpy
from bpy.props import PointerProperty, BoolProperty
from bpy.types import PropertyGroup
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
from . import ui  
from . import optode_utils
from . import sd_probe_import
from .dependencies import check_dependencies
from . import optode_connect
from . import landmark_labels
from . import optodeJSON_manualSpring as optode_json  
from .import optode_modules
from . import layered_mesh_manager
from . import redbird_runner 
from. import NeuroJSON_MeshLoader

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

class NeuroCaptainProperties(PropertyGroup):
    show_landmark_labels: BoolProperty(
        name="Show Landmark Labels",
        description="Display 10-20 landmark labels",
        default=False,
        update=lambda self, context: bpy.ops.neurocaptain.toggle_landmark_labels()
    )

def register():
    print("Registering NeuroCaptain")
    bpy.types.Scene.neurocaptain_flexible_goal_weight = bpy.props.FloatProperty(
        name="Flexible Goal Weight",
        description="Goal weight for non-anchor optodes (0.0=free to move, 1.0=fixed)",
        default=0.0,
        min=0.0,
        max=1.0,
        step=1,
        precision=2)
    
    # Register property groups first
    bpy.utils.register_class(NeuroCaptainProperties)
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
    
    landmark_labels.register()
    sd_probe_import.register()
    optode_connect.register()
    optode_modules.register()
    optode_json.register()
    ui.register()
    
    # Register scene properties
    bpy.types.Scene.niifile = PointerProperty(type=niifile)
    bpy.types.Scene.neurocaptain = PointerProperty(type=NeuroCaptainProperties)


def unregister():
    print("Unregistering NeuroCaptain")
    
    # Delete scene properties first
    del bpy.types.Scene.neurocaptain
    del bpy.types.Scene.niifile
    
    ui.unregister()
    optode_json.unregister()
    optode_connect.unregister()
    sd_probe_import.unregister()
    landmark_labels.unregister()
    optode_modules.unregister()
    
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
    bpy.utils.unregister_class(NeuroCaptainProperties)