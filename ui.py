import bpy
from .niifile import niifile
from .nii2mesh import nii2mesh
from .obj2surf import object2surf
from .stlfile import stl_file_import
from .initpoints import init_points
from .decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model

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
        bp = scene.braincapgen
        rowengine = layout.row()
        rowengine.label(text="Backend:")
        rowengine.prop(bp, "backend", expand=True)

        layout.separator()
        layout.label(text="Import head model from JNIfTI/NIfTI", icon='SHADING_SOLID')
        layout.prop(bp, "path")
        colv2m = layout.column()
        
        layout.separator()
        layout.label(text="Import head model (.stl or .off file)", icon='SHADING_SOLID')
        cols2m = layout.column()
        cols2m.operator(stl_file_import.bl_idname,icon='IMPORT')
        
        layout.separator()
        layout.label(text = "Choose from predefined headatlases", icon = 'SHADING_SOLID')
        rowmod = layout.row()
        #colmod.operator(select_model.bl_idname,icon='FILEBROWSER')
        rowmod.operator(select_model.bl_idname,text='Headmesh',icon='USER').action='ADD_HEADMESH'
        rowmod.operator(select_model.bl_idname,text='Brain1020Mesh', icon='OUTLINER_DATA_VOLUME').action='ADD_BRAIN1020MESH'

        layout.separator()
        layout.label(text = "Alter the density of the headmesh", icon = 'PREFERENCES')
        coldec = layout.column()
        coldec.operator(decimate_mesh.bl_idname,icon='MOD_DECIM')
        

        layout.separator()
        layout.label(text = "Generate Brain-Landmarks Mesh", icon = 'MOD_REMESH')
        colinit = layout.column()
        colinit.operator(init_points.bl_idname,icon='OUTLINER_OB_POINTCLOUD')

        layout.separator()
        layout.label(text = "Choose cut out shape", icon = 'MESH_CYLINDER')
        rowshape = layout.row()
        rowshape.operator(insert_shape.bl_idname,text='Circle',icon='MESH_CIRCLE').action='ADD_CYLINDER'
        rowshape.operator(insert_shape.bl_idname,text='Square', icon='MESH_PLANE').action='ADD_CUBE'
        rowshape.operator(insert_shape.bl_idname,text='Triangle',icon='MARKER').action='ADD_TRIANGLE'
        rowshape.operator(insert_shape.bl_idname,text='select the object you customized for the cut out BEFORE pressing this button',icon='RESTRICT_SELECT_OFF').action='ADD_CUSTOM'

        
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
