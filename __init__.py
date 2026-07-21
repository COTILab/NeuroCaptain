import os, sys
_libs = os.path.join(os.path.dirname(__file__), "_libs")
if os.path.isdir(_libs) and _libs not in sys.path:
    sys.path.insert(0, _libs)
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_libs)

_modules = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules")
if os.path.isdir(_modules) and _modules not in sys.path:
    sys.path.insert(0, _modules)

import bpy
from bpy.props import PointerProperty, BoolProperty
from bpy.types import PropertyGroup
from .file_import import file_import
from . import brain1020mesh as brain1020mesh_mod
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
from . import optodeJSON_blenderGoal as optode_blender_goal
from . import optode_modules
from . import layered_mesh_manager
from . import redbird_runner
from . import NeuroJSON_MeshLoader
from . import probe_variability

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


@bpy.app.handlers.persistent
def _restore_layered_mesh_on_load(dummy):
    """Re-populate layered_mesh_manager.LAYERED_MESH (in-memory only, reset
    on every Blender restart) from data saved on the head surface object, so
    a reopened .blend doesn't need the 5-layer mesh re-imported from its
    original external file just to run MMC/Redbird again."""
    layered_mesh_manager.restore_layered_mesh_from_saved_data()


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
    brain1020mesh_mod.register()
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
    optode_blender_goal.register()
    probe_variability.register()
    ui.register()
    
    # Register scene properties
    bpy.types.Scene.niifile = PointerProperty(type=niifile)
    bpy.types.Scene.neurocaptain = PointerProperty(type=NeuroCaptainProperties)

    if _restore_layered_mesh_on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_restore_layered_mesh_on_load)


def unregister():
    print("Unregistering NeuroCaptain")

    if _restore_layered_mesh_on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_layered_mesh_on_load)

    # Delete scene properties first
    del bpy.types.Scene.neurocaptain
    del bpy.types.Scene.niifile

    ui.unregister()
    probe_variability.unregister()
    optode_blender_goal.unregister()
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
    brain1020mesh_mod.unregister()
    bpy.utils.unregister_class(insert_shape)
    bpy.utils.unregister_class(decimate_mesh)
    bpy.utils.unregister_class(file_import)
    bpy.utils.unregister_class(niifile)
    bpy.utils.unregister_class(NeuroCaptainProperties)