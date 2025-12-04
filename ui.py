import bpy
from .file_import import file_import
from .brain1020mesh import brain1020mesh
from .decimate_mesh import decimate_mesh
from .shapes import insert_shape
from .headmodels import select_model
from .geonode import geo_nodes
from .dual_mesh_nc import dual_mesh_NC
from .capgen import cap_generation
from .circumference import circumference_calc
from .exportmesh import exportmesh
from .customLandmarks import customLandmarks


# Parent Panel
class NEUROCAPTAIN_PT_main_panel(bpy.types.Panel):
    bl_label = "NeuroCaptain v2024"
    bl_idname = "NEUROCAPTAIN_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NeuroCaptain"

    @classmethod
    def poll(cls, context):
        return context.mode in {"EDIT_MESH", "OBJECT", "PAINT_WEIGHT"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        bp = scene.neurocaptain

        rowengine = layout.row()
        rowengine.label(text="Backend:")
        rowengine.prop(bp, "backend", expand=True)


# Sub-panel 1: CapGen
class NEUROCAPTAIN_PT_capgen_subpanel(bpy.types.Panel):
    bl_label = "CapGen"
    bl_idname = "NEUROCAPTAIN_PT_capgen_subpanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NeuroCaptain"
    bl_parent_id = "NEUROCAPTAIN_PT_main_panel"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        layout.label(text="Import Model", icon="SHADING_SOLID")
        cols2m = layout.column()
        cols2m.operator(file_import.bl_idname, icon="IMPORT")

        layout.separator()
        layout.label(text="Choose Head Model and Landmark Geometry", icon="SHADING_SOLID")
        rowmod = layout.row()
        rowmod.operator(
            select_model.bl_idname, text="Headmesh", icon="USER"
        ).action = "ADD_HEADMESH"
        rowmod.operator(
            select_model.bl_idname, text="10-20 Mesh", icon="OUTLINER_DATA_VOLUME"
        ).action = "ADD_BRAIN1020MESH"

        layout.separator()
        layout.label(text="Generate 10-20 Landmarks Mesh", icon="SHADING_SOLID")

        rowbmesh = layout.row()

        op = rowbmesh.operator(
            brain1020mesh.bl_idname,
            text="NZ ✔️" if context.scene.nz_assigned else "NZ",
            icon="USER",
        )
        op.action = "NZ_SELECT"

        op = rowbmesh.operator(
            brain1020mesh.bl_idname,
            text="LPA ✔️" if context.scene.lpa_assigned else "LPA",
            icon="USER",
        )
        op.action = "LPA_SELECT"

        op = rowbmesh.operator(
            brain1020mesh.bl_idname,
            text="RPA ✔️" if context.scene.rpa_assigned else "RPA",
            icon="USER",
        )
        op.action = "RPA_SELECT"

        rowbmesh2 = layout.row()

        op = rowbmesh2.operator(
            brain1020mesh.bl_idname,
            text="IZ ✔️" if context.scene.iz_assigned else "IZ",
            icon="USER",
        )
        op.action = "IZ_SELECT"

        op = rowbmesh2.operator(
            brain1020mesh.bl_idname,
            text="CZ ✔️" if context.scene.cz_assigned else "CZ",
            icon="USER",
        )
        op.action = "CZ_SELECT"

        colbmesh = layout.column()
        colbmesh.operator(
            brain1020mesh.bl_idname,
            text="10-20 Mesh Generation",
            icon="OUTLINER_OB_POINTCLOUD",
        ).action = "BRAIN1020_MESH"

        layout.separator()
        layout.label(text="Define Custom Landmark Geometry", icon="SHADING_SOLID")
        rowbmesh3 = layout.row()
        rowbmesh3.operator(
            customLandmarks.bl_idname, text="Generate Custom", icon="USER"
        ).action = "CUSTOM_GENERATE"
        rowbmesh3.operator(
            customLandmarks.bl_idname, text="Label Custom", icon="USER"
        ).action = "CUSTOM_MESH"

        layout.separator()
        layout.label(text="Alter the Density of the Headmesh", icon="SHADING_SOLID")
        coldec = layout.column()
        coldec.operator(decimate_mesh.bl_idname, icon="MOD_DECIM")

        layout.separator()
        layout.label(text="Convert to Dual Mesh", icon="SHADING_SOLID")
        coldual = layout.column()
        coldual.operator(dual_mesh_NC.bl_idname, icon="SEQ_CHROMA_SCOPE")

        layout.separator()
        layout.label(text="Choose Cutout Shape", icon="SHADING_SOLID")
        rowshape = layout.row()
        rowshape.operator(
            insert_shape.bl_idname, text="Circle", icon="MESH_CIRCLE"
        ).action = "ADD_CYLINDER"
        rowshape.operator(
            insert_shape.bl_idname, text="Square", icon="MESH_PLANE"
        ).action = "ADD_CUBE"
        rowshape.operator(
            insert_shape.bl_idname, text="Triangle", icon="MARKER"
        ).action = "ADD_TRIANGLE"
        rowshape.operator(
            insert_shape.bl_idname, text="Custom", icon="RESTRICT_SELECT_OFF"
        ).action = "ADD_CUSTOM"

        layout.separator()
        layout.label(text="Integrate Landmarks with Head Surface", icon="SHADING_SOLID")
        col1020 = layout.column()
        col1020.operator(geo_nodes.bl_idname, icon="MOD_DECIM")

        layout.separator()
        layout.label(text="Cap Generation", icon="SHADING_SOLID")
        rowcap = layout.row()
        rowcap.operator(
            cap_generation.bl_idname,
            text="Reference(Nz)✔️" if context.scene.nz_assigned else "Reference (Nz)",
            icon="OUTLINER_OB_POINTCLOUD",
        ).action = "REFERENCE_POINT"
        rowcap.operator(
            cap_generation.bl_idname, text="Cutout Placement", icon="META_CUBE"
        ).action = "PLACE_CUTOUTS"
        colcap = layout.column()
        colcap.operator(
            cap_generation.bl_idname, text="Generate Cap", icon="MODIFIER_DATA"
        ).action = "BOOLEAN_CUT"

        layout.separator()
        layout.label(text="Cap Circumference Calculator", icon="SHADING_SOLID")
        colcirc = layout.column()
        colcirc.operator(circumference_calc.bl_idname, text="Cap Circumference", icon="MOD_DECIM")

        layout.separator()
        layout.label(text="Export Mesh", icon="SHADING_SOLID")
        colexp = layout.column()
        colexp.operator(exportmesh.bl_idname, text="Export Mesh", icon="MOD_DECIM")


# Sub-panel 2: Optodes
class NEUROCAPTAIN_PT_optodes_subpanel(bpy.types.Panel):
    bl_label = "Optodes"
    bl_idname = "NEUROCAPTAIN_PT_optodes_subpanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NeuroCaptain"
    bl_parent_id = "NEUROCAPTAIN_PT_main_panel"

    def draw(self, context):
        layout = self.layout

        layout.label(text="fNIRS Probe Import", icon="LIGHT_SUN")
        layout.operator("neurocaptain.import_sd_probe", text="Import SD Probe", icon="IMPORT")


def register():
    bpy.utils.register_class(NEUROCAPTAIN_PT_main_panel)
    bpy.utils.register_class(NEUROCAPTAIN_PT_capgen_subpanel)
    bpy.utils.register_class(NEUROCAPTAIN_PT_optodes_subpanel)
    bpy.types.Scene.neurocaptain_selected_action = bpy.props.StringProperty(
        name="Selected Action", description="Which landmark button is selected?", default=""
    )


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_PT_optodes_subpanel)
    bpy.utils.unregister_class(NEUROCAPTAIN_PT_capgen_subpanel)
    bpy.utils.unregister_class(NEUROCAPTAIN_PT_main_panel)
    del bpy.types.Scene.neurocaptain_selected_action