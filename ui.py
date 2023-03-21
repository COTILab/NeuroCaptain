import bpy
from .nii2mesh import nii2mesh

class BrainCapGen_UI(bpy.types.Panel):
    bl_label = 'BrainCapGen v2023'
    bl_idname = 'BRAINCAPGEN_PT_UI'
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "BrainCapGen"

    @classmethod
    def poll(self,context):
        return context.mode in {'EDIT_MESH','OBJECT','PAINT_WEIGHT'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        bp = scene.blender_photonics
        rowengine = layout.row()
        rowengine.label(text="Backend:")
        rowengine.prop(bp, "backend", expand=True)

        layout.separator()
        layout.label(text="Import head model from JNIfTI/NIfTI", icon='SHADING_SOLID')
        layout.prop(bp, "path")
        colv2m = layout.column()
        colv2m.operator(nii2mesh.bl_idname,icon='MESH_GRID')

        layout.separator()
        layout.label(text="Tutorials and Websites", icon='SHADING_SOLID')
        colurl = layout.row()
        op=colurl.operator('wm.url_open', text='Iso2Mesh',icon='URL')
        op.url='http://iso2mesh.sf.net'
        op=colurl.operator('wm.url_open', text='JMesh spec',icon='URL')
        op.url='https://github.com/NeuroJSON/jmesh/blob/master/JMesh_specification.md'
        colurl2 = layout.row()
        op=colurl2.operator('wm.url_open', text='MMC wiki',icon='URL')
        op.url='http://mcx.space/wiki/?Learn#mmc'
        op=colurl2.operator('wm.url_open', text='Brain2Mesh',icon='URL')
        op.url='http://mcx.space/brain2mesh'
        layout.label(text="Funded by NIH R01-GM114365 & U24-NS124027", icon='HEART')
