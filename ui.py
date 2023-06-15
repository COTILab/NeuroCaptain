import bpy
from .file_import import file_import
from .brain1020mesh import brain1020mesh
from .decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model
from .geonode import geo_nodes
from .dual_mesh import dual_mesh
from .capgen import cap_generation

class NeuroCaptain_UI(bpy.types.Panel):
    bl_label = 'NeuroCaptain v2023'
    bl_idname = 'NeuroCaptain_PT_UI'
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NeuroCaptain"

    @classmethod
    def poll(self,context):
        return context.mode in {'EDIT_MESH','OBJECT','PAINT_WEIGHT'}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        bp = scene.neurocaptain
        rowengine = layout.row()
        rowengine.label(text="Backend:")
        rowengine.prop(bp, "backend", expand=True)
        
        layout.separator()
        layout.label(text="Import Head Model", icon='SHADING_SOLID')
        cols2m = layout.column()
        cols2m.operator(file_import.bl_idname,icon='IMPORT')
        
        layout.separator()
        layout.label(text = "Choose from Predefined Head Atlases", icon = 'SHADING_SOLID')
        rowmod = layout.row()
        rowmod.operator(select_model.bl_idname,text='Headmesh',icon='USER').action='ADD_HEADMESH'
        rowmod.operator(select_model.bl_idname,text='Brain1020Mesh', icon='OUTLINER_DATA_VOLUME').action='ADD_BRAIN1020MESH'


        layout.separator()
        layout.label(text = "Generate Brain-Landmarks Mesh", icon = 'SHADING_SOLID')
        rowbmesh = layout.row()
        rowbmesh.operator(brain1020mesh.bl_idname,text='Select NZ',icon='USER').action='NZ_SELECT'
        rowbmesh.operator(brain1020mesh.bl_idname,text='Select LPA',icon='USER').action='LPA_SELECT'
        rowbmesh.operator(brain1020mesh.bl_idname,text='Select RPA',icon='USER').action='RPA_SELECT'
        colbmesh = layout.column()
        colbmesh.operator(brain1020mesh.bl_idname,text='Brain 10-20 Mesh',icon='OUTLINER_OB_POINTCLOUD').action='BRAIN1020_MESH'

        layout.separator()
        layout.label(text = "Alter the Density of the Headmesh", icon = 'SHADING_SOLID')
        coldec = layout.column()
        coldec.operator(decimate_mesh.bl_idname,icon='MOD_DECIM')
        

        layout.separator()
        layout.label(text = "Convert to Dual Mesh", icon = 'SHADING_SOLID')
        coldual = layout.column()
        coldual.operator(dual_mesh.bl_idname,icon='SEQ_CHROMA_SCOPE')

        layout.separator()
        layout.label(text = "Choose Cutout Shape", icon = 'SHADING_SOLID')
        rowshape = layout.row()
        rowshape.operator(insert_shape.bl_idname,text='Circle',icon='MESH_CIRCLE').action='ADD_CYLINDER'
        rowshape.operator(insert_shape.bl_idname,text='Square', icon='MESH_PLANE').action='ADD_CUBE'
        rowshape.operator(insert_shape.bl_idname,text='Triangle',icon='MARKER').action='ADD_TRIANGLE'
        rowshape.operator(insert_shape.bl_idname,text='Custom',icon='RESTRICT_SELECT_OFF').action='ADD_CUSTOM'

        layout.separator()
        layout.label(text = "Project Brain Landmarks to Head Surface", icon = 'SHADING_SOLID')
        col1020 = layout.column()
        col1020.operator(geo_nodes.bl_idname,icon='MOD_DECIM')

        layout.separator()
        layout.label(text = "Cap Generation", icon = 'SHADING_SOLID')
        rowcap = layout.row()
        rowcap.operator(cap_generation.bl_idname,text='Reference (Nz)',icon='OUTLINER_OB_POINTCLOUD').action='REFERENCE_POINT'
        rowcap.operator(cap_generation.bl_idname,text='Cutout Placement',icon='META_CUBE').action='PLACE_CUTOUTS'
        colcap = layout.column()
        colcap.operator(cap_generation.bl_idname,text='Generate NeuroCap',icon='MODIFIER_DATA').action='BOOLEAN_CUT'
        
        
    
