import bpy
from .ui import BrainCapGen_UI

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


def unregister():
    print("Unregistering BrainCapGen")
    bpy.utils.unregister_class(BrainCapGen_UI)
