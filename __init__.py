
bl_info = {
    "name": "NIRSCapGen",
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
    print("Registering BlenderPhotonics")
    bpy.utils.register_class(scene2mesh)
    bpy.utils.register_class(object2surf)
    bpy.utils.register_class(niifile)
    bpy.utils.register_class(nii2mesh)
    bpy.utils.register_class(mesh2scene)
    bpy.utils.register_class(runmmc)
    bpy.utils.register_class(BlenderPhotonics_UI)
    bpy.types.Scene.blender_photonics = PointerProperty(type=niifile)


def unregister():
    print("Unregistering BlenderPhotonics")
    bpy.utils.unregister_class(scene2mesh)
    bpy.utils.unregister_class(object2surf)
    bpy.utils.unregister_class(niifile)
    bpy.utils.unregister_class(nii2mesh)
    bpy.utils.unregister_class(mesh2scene)
    bpy.utils.unregister_class(runmmc)
    bpy.utils.unregister_class(BlenderPhotonics_UI)
    del bpy.types.Scene.blender_photonics
