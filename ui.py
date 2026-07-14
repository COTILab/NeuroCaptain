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
from .optode_utils import (
    NEUROCAPTAIN_OT_add_source,
    NEUROCAPTAIN_OT_add_detector,
    NEUROCAPTAIN_OT_move_optode,
    NEUROCAPTAIN_OT_ensure_optode_constraints,
)
from .optode_connect import (
    NEUROCAPTAIN_OT_create_optode_connections,
    NEUROCAPTAIN_OT_create_optode_connections_delaunay,
    NEUROCAPTAIN_OT_modify_spring_properties,
    NEUROCAPTAIN_OT_make_spring_stiff,
    NEUROCAPTAIN_OT_update_optode_connections,
    NEUROCAPTAIN_OT_toggle_connection_visibility,
    NEUROCAPTAIN_OT_define_anchor_optode,
    NEUROCAPTAIN_OT_undefine_anchor_optode,
    NEUROCAPTAIN_OT_export_optode_json,
    NEUROCAPTAIN_OT_import_optode_json,
    NEUROCAPTAIN_OT_rigid_rotate_optodes,
)
from .landmark_labels import (
    NEUROCAPTAIN_OT_display_landmark_labels,
    NEUROCAPTAIN_OT_toggle_landmark_labels,
)
from .optodeJSON_blenderGoal import (
    NEUROCAPTAIN_OT_import_optode_json_blender_goal,
    NEUROCAPTAIN_OT_relax_probe_manual,
)
from .optode_modules import (
    NEUROCAPTAIN_UL_optode_modules,
    NEUROCAPTAIN_OT_create_optode_module,
    NEUROCAPTAIN_OT_delete_optode_module,
    NEUROCAPTAIN_OT_move_module_group,
)
from . import lightsim_neurocaptain as lightsim
from . import layered_mesh_manager as lmm
from . import NeuroJSON_MeshLoader as njloader
from . import schematic_2d
from . import probe_variability


# ============================================================================
# SETTINGS
# ============================================================================

class NeuroCaptainSettings(bpy.types.PropertyGroup):
    active_tab: bpy.props.EnumProperty(
        name="Active Tab",
        items=[
            ("CAPGEN",   "CapGen",       "Cap generation tools"),
            ("OPTODES",  "Optodes",      "Optode placement and connections"),
            ("LIGHTSIM", "Light Sim",    "Forward light simulation"),
            ("DEPS",     "Dependencies", "Dependency management"),
        ],
        default="CAPGEN",
    )
    sd_max_distance: bpy.props.FloatProperty(
        name="Max SD Distance (mm)", default=60.0, min=10.0, max=150.0, step=5, precision=1
    )
    smooth_iterations: bpy.props.IntProperty(
        name="Smoothing Iterations", default=5, min=0, max=20
    )
    # MMC
    mmc_nphoton: bpy.props.IntProperty(
        name="Photons", default=1000, min=100, max=10000000, step=1000
    )
    mmc_use_gpu: bpy.props.BoolProperty(name="Use GPU", default=True)
    mmc_gpu_id:  bpy.props.StringProperty(name="GPU ID", default="01")
    # Sensitivity colormap range (shared by MMC and Redbird)
    viz_custom_range: bpy.props.BoolProperty(
        name="Custom Colormap Range",
        description="Override the auto min/max range for the sensitivity colormap",
        default=False,
    )
    viz_vmin: bpy.props.FloatProperty(
        name="Min (log10)",
        description="Minimum log10 sensitivity value mapped to the lowest colormap color",
        default=-15.0, min=-30.0, max=0.0, precision=1,
    )
    viz_vmax: bpy.props.FloatProperty(
        name="Max (log10)",
        description="Maximum log10 sensitivity value mapped to the highest colormap color",
        default=-5.0, min=-30.0, max=0.0, precision=1,
    )
    # Redbird
    redbird_mode: bpy.props.EnumProperty(
        name="Mode",
        items=[('CW', "Continuous Wave", ""), ('FD', "Frequency Domain", "")],
        default='CW',
    )
    redbird_frequency:          bpy.props.FloatProperty(name="Frequency (MHz)",       default=70.0, min=0.0,  max=1000.0, step=10,  precision=1)
    redbird_crop_margin:         bpy.props.FloatProperty(name="Crop Margin (mm)",       default=10.0, min=5.0,  max=50.0,   step=5,   precision=1,
                                    description="mm to extend beyond optode bounding box on all sides")
    redbird_min_depth:           bpy.props.FloatProperty(name="Min Depth (mm)",        default=2.0,  min=0.5,  max=10.0,   step=0.5, precision=1)
    redbird_max_iter:    bpy.props.IntProperty(  name="Max Iterations",   default=10,   min=1,   max=10000,
                             description="Maximum CG solver iterations for the FEM forward solve")
    redbird_lambda:      bpy.props.FloatProperty(name="Regularization \u03bb", default=1e-6, min=1e-12, max=1.0, precision=8,
                             description="Tikhonov regularization applied to the FEM stiffness matrix (stabilises the solver)")
    # Schematic
    schematic_landmark_tier: bpy.props.EnumProperty(
        name="Landmarks",
        items=[
            ("NONE", "None",  ""),
            ("1020", "10-20", ""),
            ("1010", "10-10", ""),
            ("105",  "10-5",  ""),
        ],
        default="NONE",
        update=lambda self, ctx: bpy.ops.neurocaptain.refresh_2d_schematic(),
    )
    schematic_show_channels: bpy.props.BoolProperty(
        name="Channels", default=True,
        update=lambda self, ctx: bpy.ops.neurocaptain.refresh_2d_schematic(),
    )
    schematic_show_lm_labels: bpy.props.BoolProperty(
        name="Landmark Labels", default=False,
        update=lambda self, ctx: bpy.ops.neurocaptain.refresh_2d_schematic(),
    )


# ============================================================================
# LAYERED MESH IMPORT
# ============================================================================

class NEUROCAPTAIN_OT_import_layered_mesh(bpy.types.Operator):
    """Import 5-Layer Head Mesh for Forward Simulation"""
    bl_idname = "neurocaptain.import_layered_mesh"
    bl_label  = "Import 5-Layer Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    filepath:    bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mat", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        deps = lmm.check_dependencies()
        if not deps['all_available']:
            missing = [k for k in ('iso2mesh', 'redbirdpy') if not deps[k]]
            self.report({'ERROR'}, f"Missing dependencies: {', '.join(missing)}")
            return {'CANCELLED'}
        try:
            result = lmm.import_layered_head_model(self.filepath, reference_obj_name='headmesh')
            self.report({'INFO'} if result['success'] else {'ERROR'}, result['message'])
            return {'FINISHED'} if result['success'] else {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


# ── Flexible (N-layer) mesh import ──────────────────────────────────────────

_LAYER_ROLE_ITEMS = [
    ('scalp',        "Scalp",        ""),
    ('skull',        "Skull",        ""),
    ('csf',          "CSF",          ""),
    ('gray_matter',  "Gray Matter",  ""),
    ('white_matter', "White Matter", ""),
    ('other',        "Other",        ""),
    ('custom',       "Custom...",    ""),
]
_MAX_FLEXIBLE_LAYERS = 8


class NEUROCAPTAIN_OT_import_layered_mesh_flexible(bpy.types.Operator):
    """Import a layered head model with any number of tissue layers (fewer
    than the standard 5, or more). Looks for layer names in the mesh file or
    a sidecar JSON; prompts you to define them if none are found"""
    bl_idname = "neurocaptain.import_layered_mesh_flexible"
    bl_label  = "Import Layered Mesh (Any Layer Count)"
    bl_options = {'REGISTER', 'UNDO'}

    filepath:    bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.mat;*.jmsh;*.bmsh;*.json", options={'HIDDEN'})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        if not lmm.ISO2MESH_AVAILABLE:
            self.report({'ERROR'}, "Missing dependency: iso2mesh")
            return {'CANCELLED'}

        try:
            source = lmm.load_layered_mesh_source(self.filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to read mesh: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}

        if source['layer_definitions'] is not None:
            try:
                result = lmm.import_layered_head_model_flexible(
                    self.filepath, reference_obj_name='headmesh',
                    layer_definitions=source['layer_definitions'],
                )
                self.report({'INFO'} if result['success'] else {'ERROR'}, result['message'])
                return {'FINISHED'} if result['success'] else {'CANCELLED'}
            except Exception as e:
                self.report({'ERROR'}, f"Import failed: {e}")
                import traceback; traceback.print_exc()
                return {'CANCELLED'}

        # No layer names found anywhere in/around the file -> ask the user
        num_layers = len(source['unique_labels'])
        if num_layers > _MAX_FLEXIBLE_LAYERS:
            self.report({'ERROR'}, f"{num_layers} layers detected, exceeds the {_MAX_FLEXIBLE_LAYERS} supported by the naming prompt")
            return {'CANCELLED'}

        bpy.ops.neurocaptain.define_layers(
            'INVOKE_DEFAULT',
            mesh_path=self.filepath,
            reference_obj_name='headmesh',
            label_ids=",".join(str(i) for i in source['unique_labels']),
        )
        return {'FINISHED'}


class NEUROCAPTAIN_OT_define_layers(bpy.types.Operator):
    """Assign a name and tissue role to each detected layer, then import"""
    bl_idname = "neurocaptain.define_layers"
    bl_label  = "Define Mesh Layers"
    bl_options = {'REGISTER', 'UNDO'}

    mesh_path:          bpy.props.StringProperty(subtype="FILE_PATH", options={'HIDDEN'})
    reference_obj_name: bpy.props.StringProperty(default="headmesh", options={'HIDDEN'})
    label_ids:          bpy.props.StringProperty(options={'HIDDEN'})  # comma-separated tissue label ids

    __annotations__ = dict(__annotations__)
    for _i in range(_MAX_FLEXIBLE_LAYERS):
        __annotations__[f'name_{_i}'] = bpy.props.StringProperty(name="Name", default="")
        __annotations__[f'role_{_i}'] = bpy.props.EnumProperty(name="Role", items=_LAYER_ROLE_ITEMS, default='other')
        __annotations__[f'custom_{_i}'] = bpy.props.StringProperty(name="Custom Role", default="")
    del _i

    def _label_ids(self):
        return [int(x) for x in self.label_ids.split(',') if x]

    def invoke(self, context, event):
        # Pre-fill names/roles from default guesses so the user only edits what's wrong
        for i, label_id in enumerate(self._label_ids()):
            setattr(self, f'name_{i}', f"Layer {label_id}")
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        layout = self.layout
        ids = self._label_ids()
        layout.label(text=f"No layer names found — detected {len(ids)} tissue label(s):", icon='INFO')
        for i, label_id in enumerate(ids):
            box = layout.box()
            row = box.row(align=True)
            row.label(text=f"Tissue {label_id}:")
            row.prop(self, f'name_{i}', text="")
            row2 = box.row(align=True)
            row2.prop(self, f'role_{i}', text="Role")
            if getattr(self, f'role_{i}') == 'custom':
                row2.prop(self, f'custom_{i}', text="")

    def execute(self, context):
        role_labels = {identifier: label for identifier, label, _ in _LAYER_ROLE_ITEMS}
        layer_definitions = {}
        for i, label_id in enumerate(self._label_ids()):
            role = getattr(self, f'role_{i}')
            name = getattr(self, f'name_{i}').strip()
            if role == 'custom':
                custom = getattr(self, f'custom_{i}').strip()
                role_key = custom.lower().replace(' ', '_') if custom else 'other'
                if not name:
                    name = custom or f"Layer {label_id}"
            else:
                role_key = role
                if not name:
                    name = role_labels[role] if role != 'other' else f"Layer {label_id}"
            layer_definitions[label_id] = {'name': name, 'role': role_key}

        try:
            result = lmm.import_layered_head_model_flexible(
                self.mesh_path, reference_obj_name=self.reference_obj_name,
                layer_definitions=layer_definitions,
            )
            self.report({'INFO'} if result['success'] else {'ERROR'}, result['message'])
            return {'FINISHED'} if result['success'] else {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Import failed: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


# ============================================================================
# MMC OPERATORS
# ============================================================================

class NEUROCAPTAIN_OT_setup_mmc(bpy.types.Operator):
    """Configure MMC Simulation Parameters"""
    bl_idname = "neurocaptain.setup_mmc"
    bl_label  = "Configure MMC"
    bl_options = {'REGISTER', 'UNDO'}

    # Optical properties per layer (5 layers × 4 params)
    _LAYER_NAMES = ["Scalp", "Skull", "CSF", "Gray Matter", "White Matter"]
    _PARAM_LABELS = {
        'mua': 'Absorb (mua)',
        'mus': 'Scatter (mus)',
        'g':   'Anisotropy (g)',
        'n':   'Ref Index (n)',
    }
    _LAYER_DEFAULTS = {
        1: {'mua': 0.019, 'mus': 7.8,   'g': 0.89, 'n': 1.37},
        2: {'mua': 0.019, 'mus': 7.8,   'g': 0.89, 'n': 1.37},
        3: {'mua': 0.004, 'mus': 0.009, 'g': 0.89, 'n': 1.37},
        4: {'mua': 0.02,  'mus': 9.0,   'g': 0.89, 'n': 1.37},
        5: {'mua': 0.08,  'mus': 40.9,  'g': 0.84, 'n': 1.37},
    }
    __annotations__ = {}
    for _l, _lname in enumerate(_LAYER_NAMES, 1):
        for _p in ('mua', 'mus', 'g', 'n'):
            __annotations__[f'layer{_l}_{_p}'] = bpy.props.FloatProperty(
                name=_PARAM_LABELS[_p], default=_LAYER_DEFAULTS[_l][_p],
                min=0.0, max=100.0, precision=4,
            )

    def invoke(self, context, event):
        if not lmm.is_mesh_loaded():
            self.report({'ERROR'}, "Import 5-layer mesh first.")
            return {'CANCELLED'}
        # Pre-fill from any previously stored MMC optical properties
        import json
        stored = context.scene.get('mmc_optical_properties')
        if stored:
            try:
                saved = json.loads(stored)
                for l in range(1, 6):
                    for p in ('mua', 'mus', 'g', 'n'):
                        val = saved.get(str(l), {}).get(p)
                        if val is not None:
                            setattr(self, f'layer{l}_{p}', val)
            except Exception:
                pass
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        nc = context.scene.neurocaptain_settings

        box = layout.box()
        box.label(text="MMC Parameters", icon='SETTINGS')
        box.prop(nc, "mmc_nphoton")
        row = box.row(align=True)
        row.prop(nc, "mmc_use_gpu")
        if nc.mmc_use_gpu:
            row.prop(nc, "mmc_gpu_id", text="GPU ID")
        else:
            row.label(text="CPU mode (no GPU required)", icon='INFO')
        box.prop(nc, "sd_max_distance")
        box.prop(nc, "smooth_iterations")

        layout.label(text="Optical Properties", icon='LIGHT_SUN')
        for i, name in enumerate(["Scalp", "Skull", "CSF", "Gray Matter", "White Matter"], 1):
            box = layout.box()
            box.label(text=f"Layer {i}: {name}")
            r = box.row(align=True)
            for p in ('mua', 'mus', 'g', 'n'):
                r.prop(self, f'layer{i}_{p}')

    def execute(self, context):
        import json
        optical_properties = {
            str(l): {p: getattr(self, f'layer{l}_{p}') for p in ('mua', 'mus', 'g', 'n')}
            for l in range(1, 6)
        }
        context.scene['mmc_optical_properties'] = json.dumps(optical_properties)
        self.report({'INFO'}, "MMC configured.")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_run_mmc(bpy.types.Operator):
    """Run MMC Simulation"""
    bl_idname = "neurocaptain.run_mmc"
    bl_label  = "Run MMC"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not lmm.is_mesh_loaded():
            self.report({'ERROR'}, "Import 5-layer mesh first.")
            return {'CANCELLED'}
        try:
            import json
            nc = context.scene.neurocaptain_settings

            # Load custom optical properties if they were configured
            optical_properties = None
            stored = context.scene.get('mmc_optical_properties')
            if stored:
                try:
                    raw = json.loads(stored)
                    optical_properties = {int(k): v for k, v in raw.items()}
                except Exception:
                    pass

            data = lightsim.load_mesh_and_register_optodes(
                mesh_path=lmm.LAYERED_MESH.mesh_path,
                optical_properties=optical_properties,
            )
            results = lightsim.run_sensitivity_mmc(data, {
                'nphoton':        nc.mmc_nphoton,
                'use_gpu':        nc.mmc_use_gpu,
                'gpuid':          nc.mmc_gpu_id if nc.mmc_use_gpu else None,
                'sd_max_distance': nc.sd_max_distance,
            })
            if results:
                vmin = nc.viz_vmin if nc.viz_custom_range else None
                vmax = nc.viz_vmax if nc.viz_custom_range else None
                lightsim.visualize_on_cortex(results, data,
                                             smooth_iterations=nc.smooth_iterations,
                                             vmin_override=vmin, vmax_override=vmax)
                self.report({'INFO'}, "MMC complete — sensitivity on cortex.")
                return {'FINISHED'}
            self.report({'ERROR'}, "MMC failed — see console.")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"MMC error: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


# ============================================================================
# REDBIRD OPERATORS
# ============================================================================

class NEUROCAPTAIN_OT_setup_redbird(bpy.types.Operator):
    """Configure Redbird Simulation Parameters"""
    bl_idname = "neurocaptain.setup_redbird"
    bl_label  = "Configure Redbird"
    bl_options = {'REGISTER', 'UNDO'}

    # Optical properties per layer (5 layers × 4 params)
    _LAYER_DEFAULTS = {
        1: {'mua': 0.019, 'mus': 7.8,   'g': 0.89, 'n': 1.37},
        2: {'mua': 0.019, 'mus': 7.8,   'g': 0.89, 'n': 1.37},
        3: {'mua': 0.004, 'mus': 0.009, 'g': 0.89, 'n': 1.37},
        4: {'mua': 0.02,  'mus': 9.0,   'g': 0.89, 'n': 1.37},
        5: {'mua': 0.08,  'mus': 40.9,  'g': 0.84, 'n': 1.37},
    }
    __annotations__ = {}
    for _l in range(1, 6):
        for _p in ('mua', 'mus', 'g', 'n'):
            __annotations__[f'layer{_l}_{_p}'] = bpy.props.FloatProperty(
                name=f'L{_l} {_p}', default=_LAYER_DEFAULTS[_l][_p],
                min=0.0, max=100.0, precision=4,
            )

    def invoke(self, context, event):
        if not lmm.is_mesh_loaded():
            self.report({'ERROR'}, "Import 5-layer mesh first.")
            return {'CANCELLED'}
        for layer in range(1, 6):
            if layer in lmm.REDBIRD_CONFIG.optical_properties:
                for p, v in lmm.REDBIRD_CONFIG.optical_properties[layer].items():
                    if hasattr(self, f'layer{layer}_{p}'):
                        setattr(self, f'layer{layer}_{p}', v)
        return context.window_manager.invoke_props_dialog(self, width=400)

    def draw(self, context):
        layout = self.layout
        nc = context.scene.neurocaptain_settings

        # ── Simulation parameters ────────────────────────────────────────
        box = layout.box()
        box.label(text="Simulation Parameters", icon='SETTINGS')
        box.prop(nc, "redbird_mode")
        if nc.redbird_mode == 'FD':
            box.prop(nc, "redbird_frequency")
        for p in ("sd_max_distance", "redbird_crop_margin", "redbird_min_depth"):
            box.prop(nc, p)

        # ── Solver settings ──────────────────────────────────────────────
        box = layout.box()
        box.label(text="Solver Settings", icon='MODIFIER')
        row = box.row(align=True)
        row.prop(nc, "redbird_max_iter")
        row.prop(nc, "redbird_lambda")
        box.prop(nc, "smooth_iterations")

        # ── Optical properties ───────────────────────────────────────────
        layout.label(text="Optical Properties", icon='LIGHT_SUN')
        for i, name in enumerate(["Scalp", "Skull", "CSF", "Gray Matter", "White Matter"], 1):
            box = layout.box()
            box.label(text=f"Layer {i}: {name}")
            r = box.row(align=True)
            for p in ('mua', 'mus', 'g', 'n'):
                r.prop(self, f'layer{i}_{p}')

    def execute(self, context):
        nc = context.scene.neurocaptain_settings
        optical_properties = {
            l: {p: getattr(self, f'layer{l}_{p}') for p in ('mua', 'mus', 'g', 'n')}
            for l in range(1, 6)
        }
        result = lmm.setup_redbird_config(
            mode=nc.redbird_mode, frequency=nc.redbird_frequency,
            sd_max_distance=nc.sd_max_distance,
            crop_margin=nc.redbird_crop_margin,
            min_depth=nc.redbird_min_depth, smooth_iterations=nc.smooth_iterations,
            max_iter=nc.redbird_max_iter, regularization_lambda=nc.redbird_lambda,
            optical_properties=optical_properties,
        )
        self.report({'INFO'} if result['success'] else {'ERROR'},
                    "Redbird configured." if result['success'] else "Configuration failed.")
        return {'FINISHED'} if result['success'] else {'CANCELLED'}


class NEUROCAPTAIN_OT_run_redbird(bpy.types.Operator):
    """Run Redbird Simulation"""
    bl_idname = "neurocaptain.run_redbird"
    bl_label  = "Run Redbird"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if not lmm.is_mesh_loaded():
            self.report({'ERROR'}, "Import 5-layer mesh first.")
            return {'CANCELLED'}
        if not lmm.check_dependencies()['redbirdpy']:
            self.report({'ERROR'}, "Install redbirdpy first.")
            return {'CANCELLED'}
        for col, label in (('Sources', 'sources'), ('Detectors', 'detectors')):
            if col not in bpy.data.collections or not bpy.data.collections[col].objects:
                self.report({'ERROR'}, f"No {label} found.")
                return {'CANCELLED'}
        try:
            nc = context.scene.neurocaptain_settings
            lmm.setup_redbird_config(
                mode=nc.redbird_mode, frequency=nc.redbird_frequency,
                sd_max_distance=nc.sd_max_distance,
                crop_margin=nc.redbird_crop_margin,
                min_depth=nc.redbird_min_depth, smooth_iterations=nc.smooth_iterations,
                max_iter=nc.redbird_max_iter, regularization_lambda=nc.redbird_lambda,
            )
            from . import redbird_runner
            result = redbird_runner.run_redbird_simulation()
            self.report({'INFO'} if result['success'] else {'ERROR'},
                        "Redbird complete!" if result['success'] else result['message'])
            return {'FINISHED'} if result['success'] else {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Redbird error: {e}")
            import traceback; traceback.print_exc()
            return {'CANCELLED'}


class NEUROCAPTAIN_OT_show_mesh_info(bpy.types.Operator):
    """Print mesh info to console"""
    bl_idname = "neurocaptain.show_mesh_info"
    bl_label  = "Show Mesh Info"
    bl_options = {'REGISTER'}

    def execute(self, context):
        if lmm.is_mesh_loaded():
            print(lmm.get_layered_mesh_info())
            self.report({'INFO'}, "Mesh info printed to console.")
        else:
            self.report({'WARNING'}, "No mesh loaded.")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_show_redbird_config(bpy.types.Operator):
    """Print Redbird config to console"""
    bl_idname = "neurocaptain.show_redbird_config"
    bl_label  = "Show Redbird Config"
    bl_options = {'REGISTER'}

    def execute(self, context):
        print(lmm.get_redbird_config_info())
        self.report({'INFO'}, "Redbird config printed to console.")
        return {'FINISHED'}


# ============================================================================
# PANELS
# ============================================================================

class NEUROCAPTAIN_PT_main_panel(bpy.types.Panel):
    bl_label      = "NeuroCaptain v2024"
    bl_idname     = "NEUROCAPTAIN_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NeuroCaptain"

    @classmethod
    def poll(cls, context):
        return context.mode in {"EDIT_MESH", "OBJECT", "PAINT_WEIGHT"}

    def draw(self, context):
        layout = self.layout
        try:
            layout.row().prop(context.scene.neurocaptain_settings, "active_tab", expand=True)
        except AttributeError:
            layout.label(text="Error: restart Blender.", icon="ERROR")


def _tab_poll(tab):
    """Factory for subpanel poll methods."""
    @classmethod
    def poll(cls, context):
        try:
            return context.scene.neurocaptain_settings.active_tab == tab
        except AttributeError:
            return False
    return poll


# ── CapGen ───────────────────────────────────────────────────────────────────

class NEUROCAPTAIN_PT_capgen_subpanel(bpy.types.Panel):
    bl_label      = "CapGen"
    bl_idname     = "NEUROCAPTAIN_PT_capgen_subpanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NeuroCaptain"
    bl_parent_id  = "NEUROCAPTAIN_PT_main_panel"
    poll          = _tab_poll("CAPGEN")

    def draw(self, context):
        layout = self.layout
        scene  = context.scene
        props  = scene.neurocaptain

        layout.label(text="Import Model", icon="SHADING_SOLID")
        layout.operator(file_import.bl_idname, icon="IMPORT")
        layout.separator()
        njloader.draw_neurojson_browser(layout, context)

        layout.separator()
        layout.label(text="Head Model & Landmark Geometry", icon="SHADING_SOLID")
        row = layout.row()
        row.operator(select_model.bl_idname, text="Headmesh",   icon="USER").action = "ADD_HEADMESH"
        row.operator(select_model.bl_idname, text="10-20 Mesh", icon="OUTLINER_DATA_VOLUME").action = "ADD_BRAIN1020MESH"

        layout.separator()
        layout.label(text="Generate 10-20 Landmarks", icon="SHADING_SOLID")
        row1 = layout.row()
        for action, label, assigned in [
            ("NZ_SELECT",  "NZ",  scene.nz_assigned),
            ("LPA_SELECT", "LPA", scene.lpa_assigned),
            ("RPA_SELECT", "RPA", scene.rpa_assigned),
        ]:
            row1.operator(brain1020mesh.bl_idname,
                          text=f"{label} (check)" if assigned else label,
                          icon="USER").action = action
        row2 = layout.row()
        for action, label, assigned in [
            ("IZ_SELECT", "IZ", scene.iz_assigned),
            ("CZ_SELECT", "CZ", scene.cz_assigned),
        ]:
            row2.operator(brain1020mesh.bl_idname,
                          text=f"{label} (check)" if assigned else label,
                          icon="USER").action = action
        layout.operator(brain1020mesh.bl_idname,
                        text="10-20 Mesh Generation",
                        icon="OUTLINER_OB_POINTCLOUD").action = "BRAIN1020_MESH"
        layout.prop(scene, "save_landmark_file")
        if scene.save_landmark_file:
            layout.prop(scene, "save_landmark_filepath")

        layout.label(text="Landmark Labels", icon="FONT_DATA")
        layout.operator("neurocaptain.display_landmark_labels", text="Generate 10-20 Labels", icon="ADD")
        if "Landmark_Labels" in bpy.data.collections:
            layout.prop(props, "show_landmark_labels", text="Show Labels", toggle=True)

        layout.separator()
        layout.label(text="Custom Landmark Geometry", icon="SHADING_SOLID")
        row = layout.row()
        row.operator(customLandmarks.bl_idname, text="Generate Custom",   icon="USER").action = "CUSTOM_GENERATE"
        row.operator(customLandmarks.bl_idname, text="Label Custom",      icon="USER").action = "CUSTOM_MESH"
        row.operator(customLandmarks.bl_idname, text="Optode Landmarks",  icon="LIGHTPROBE_VOLUME" if bpy.app.version >= (4, 2, 0) else "LIGHTPROBE_GRID").action = "OPTODE_LANDMARKS"

        layout.separator()
        layout.label(text="Mesh Tools", icon="SHADING_SOLID")
        layout.operator(decimate_mesh.bl_idname, icon="MOD_DECIM")
        layout.operator(dual_mesh_NC.bl_idname,  icon="SEQ_CHROMA_SCOPE")

        layout.separator()
        layout.label(text="Cutout Shape", icon="SHADING_SOLID")
        row = layout.row()
        row.operator(insert_shape.bl_idname, text="Circle",   icon="MESH_CIRCLE").action   = "ADD_CYLINDER"
        row.operator(insert_shape.bl_idname, text="Square",   icon="MESH_PLANE").action    = "ADD_CUBE"
        row.operator(insert_shape.bl_idname, text="Triangle", icon="MARKER").action        = "ADD_TRIANGLE"
        row.operator(insert_shape.bl_idname, text="Custom",   icon="RESTRICT_SELECT_OFF").action = "ADD_CUSTOM"

        layout.separator()
        layout.label(text="Integrate Landmarks with Surface", icon="SHADING_SOLID")
        layout.operator(geo_nodes.bl_idname, icon="MOD_DECIM")

        layout.separator()
        layout.label(text="Cap Generation", icon="SHADING_SOLID")
        row = layout.row()
        row.operator(cap_generation.bl_idname,
                     text="Reference (Nz) (check)" if scene.nz_assigned else "Reference (Nz)",
                     icon="OUTLINER_OB_POINTCLOUD").action = "REFERENCE_POINT"
        row.operator(cap_generation.bl_idname, text="Cutout Placement", icon="META_CUBE").action = "PLACE_CUTOUTS"
        layout.operator(cap_generation.bl_idname, text="Generate Cap", icon="MODIFIER_DATA").action = "BOOLEAN_CUT"

        layout.separator()
        layout.label(text="Utilities", icon="SHADING_SOLID")
        layout.operator(circumference_calc.bl_idname, text="Cap Circumference", icon="MOD_DECIM")
        layout.operator(exportmesh.bl_idname,         text="Export Mesh",       icon="MOD_DECIM")


# ── Optodes ──────────────────────────────────────────────────────────────────

class NEUROCAPTAIN_PT_optodes_subpanel(bpy.types.Panel):
    bl_label      = "Optodes"
    bl_idname     = "NEUROCAPTAIN_PT_optodes_subpanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NeuroCaptain"
    bl_parent_id  = "NEUROCAPTAIN_PT_main_panel"
    poll          = _tab_poll("OPTODES")

    def draw(self, context):
        layout = self.layout
        nc     = context.scene.neurocaptain_settings
        props  = context.scene.neurocaptain

        # ── Placement ────────────────────────────────────────────────────
        layout.label(text="Placement", icon="EMPTY_SINGLE_ARROW")
        row = layout.row()
        row.operator("neurocaptain.add_source",   text="Add Source",   icon="LIGHT_SUN")
        row.operator("neurocaptain.add_detector", text="Add Detector", icon="RADIOBUT_OFF")
        row = layout.row()
        row.operator("neurocaptain.move_optode",
                     text="Move Selected", icon="RESTRICT_SELECT_OFF")
        row.operator("neurocaptain.ensure_optode_constraints",
                     text="Constrain to Head", icon="CONSTRAINT")

        # ── Landmark labels ──────────────────────────────────────────────
        layout.separator()
        layout.label(text="Landmark Labels", icon="FONT_DATA")
        layout.operator("neurocaptain.display_landmark_labels", text="Generate 10-20 Labels", icon="ADD")
        if "Landmark_Labels" in bpy.data.collections:
            layout.prop(props, "show_landmark_labels", text="Show Labels", toggle=True)

        # ── Probe import ─────────────────────────────────────────────────
        layout.separator()
        layout.label(text="Probe Import", icon="IMPORT")
        row = layout.row()
        row.operator("neurocaptain.import_sd_probe",                 text="Import SD",   icon="FILEBROWSER")
        row.operator("neurocaptain.import_optode_json_blender_goal", text="Import JSON", icon="IMPORT")

        # Spring relaxation status
        if "Optode_Connections" in bpy.data.objects:
            conn_obj = bpy.data.objects["Optode_Connections"]
            if "optode_names_ordered" in conn_obj:
                layout.label(text=f"✓ Probe loaded ({len(conn_obj['optode_names_ordered'])} optodes)", icon='CHECKMARK')
        layout.operator("neurocaptain.relax_probe_manual", text="Relax Probe", icon="MOD_CLOTH")

        # ── Connections ──────────────────────────────────────────────────
        layout.separator()
        layout.label(text="Connections", icon="CONSTRAINT")
        row = layout.row()
        row.operator("neurocaptain.create_optode_connections",          text="By Distance", icon="OUTLINER_OB_FORCE_FIELD")
        row.operator("neurocaptain.create_optode_connections_delaunay", text="Delaunay",    icon="MESH_DATA")
        row = layout.row()
        row.operator("neurocaptain.update_optode_connections",    text="Update",          icon="FILE_REFRESH")
        row.operator("neurocaptain.toggle_connection_visibility", text="Toggle Visible",  icon="HIDE_OFF")

        # ── Spring physics ───────────────────────────────────────────────
        layout.separator()
        layout.label(text="Spring Physics (Edit Mode)", icon="FORCE_HARMONIC")
        row = layout.row()
        row.operator("neurocaptain.modify_spring_properties", text="Flexible", icon="FORCE_HARMONIC")
        row.operator("neurocaptain.make_spring_stiff",        text="Stiff",    icon="RIGID_BODY_CONSTRAINT")

        layout.separator()
        layout.label(text="Anchors", icon="PINNED")
        row = layout.row()
        row.operator("neurocaptain.define_anchor_optode",   text="Set Anchor",    icon="PINNED")
        row.operator("neurocaptain.undefine_anchor_optode", text="Remove Anchor", icon="UNPINNED")

        # ── Modules ──────────────────────────────────────────────────────
        layout.separator()
        layout.label(text="Modules", icon="GROUP")
        row = layout.row()
        row.template_list("NEUROCAPTAIN_UL_optode_modules", "",
                          context.scene, "neurocaptain_optode_modules",
                          context.scene, "neurocaptain_optode_modules_index", rows=3)
        col = row.column(align=True)
        col.operator("neurocaptain.create_optode_module", icon='ADD',    text="")
        col.operator("neurocaptain.delete_optode_module", icon='REMOVE', text="")

        # ── Transform / Export ───────────────────────────────────────────
        layout.separator()
        layout.operator("neurocaptain.rigid_rotate_optodes",  text="Rigid Rotate Selected", icon="CON_ROTLIKE")
        layout.operator("neurocaptain.export_optode_json",    text="Export Optode Config",     icon="EXPORT")

        # ── 2D Schematic ─────────────────────────────────────────────────
        layout.separator()
        layout.label(text="2D Schematic", icon="IMAGE_DATA")
        box  = layout.box()
        opts = schematic_2d.collect_optodes()
        if opts:
            n_src = sum(1 for o in opts if o["type"] == "source")
            n_det = sum(1 for o in opts if o["type"] == "detector")
            box.label(text=f"Scene: {n_src} sources, {n_det} detectors", icon="CHECKMARK")
            box.operator("neurocaptain.open_2d_schematic", text="Open Schematic", icon="IMAGE_DATA")
        else:
            box.label(text="No scene optodes — will import JSON", icon="IMPORT")
            box.operator("neurocaptain.open_2d_schematic", text="Import Probe & Visualize", icon="FILEBROWSER")
        box.operator("neurocaptain.refresh_2d_schematic", text="Refresh", icon="FILE_REFRESH")
        col = box.column(align=True)
        col.label(text="Landmark overlay:")
        row = col.row(align=True)
        for tier in ("NONE", "1020", "1010", "105"):
            row.prop_enum(nc, "schematic_landmark_tier", tier)
        col.prop(nc, "schematic_show_lm_labels")
        row = col.row(align=True)
        row.prop(nc, "schematic_show_channels", toggle=True)
        if nc.schematic_show_channels:
            row.prop(nc, "sd_max_distance", text="Max mm")

        # ── Probe Variability Analysis ──────────────────────────────────
        layout.separator()
        layout.label(text="Probe Variability Analysis", icon="FORCE_VORTEX")
        box = layout.box()
        box.label(text="Individual Subject", icon="USER")
        box.operator("neurocaptain.export_subject_json",
                     text="Export Subject Probe Evaluation", icon="EXPORT")

        box.separator()
        box.label(text="Group Analysis", icon="COMMUNITY")
        box.operator("neurocaptain.run_variability",
                     text="Run Group Analysis", icon="PLAY")
        box.operator("neurocaptain.variability_clear",
                     text="Clear Results", icon="TRASH")


# ── Light Simulation ─────────────────────────────────────────────────────────

class NEUROCAPTAIN_PT_lightsim_subpanel(bpy.types.Panel):
    bl_label      = "Light Simulation"
    bl_idname     = "NEUROCAPTAIN_PT_lightsim_subpanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NeuroCaptain"
    bl_parent_id  = "NEUROCAPTAIN_PT_main_panel"
    poll          = _tab_poll("LIGHTSIM")

    def draw(self, context):
        layout = self.layout
        nc     = context.scene.neurocaptain_settings

        # ── 5-Layer mesh ─────────────────────────────────────────────────
        layout.label(text="5-Layer Mesh", icon="MESH_DATA")
        if lmm.is_mesh_loaded():
            box = layout.box()
            box.label(text="✓ Mesh Loaded", icon='CHECKMARK')
            row = box.row()
            row.operator("neurocaptain.show_mesh_info",      text="Info",           icon="INFO")
            row.operator("neurocaptain.import_layered_mesh", text="Load Different", icon="FILE_REFRESH")
        else:
            layout.operator("neurocaptain.import_layered_mesh", text="From File (.mat)", icon="IMPORT")
        layout.operator("neurocaptain.import_layered_mesh_flexible",
                         text="From File (any layer count)", icon="IMPORT")

        layout.separator()
        njloader.draw_neurojson_browser(layout, context)

        # ── Analysis settings ────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="Analysis Settings", icon="SETTINGS")
        box.prop(nc, "sd_max_distance")
        box.prop(nc, "smooth_iterations")

        # ── MMC ──────────────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="MMC (Monte Carlo)", icon="LIGHTPROBE_VOLUME" if bpy.app.version >= (4, 2, 0) else "LIGHTPROBE_GRID")
        if lmm.is_mesh_loaded():
            info_row = box.row()
            info_row.label(text=f"✓ Mesh ready ({lmm.LAYERED_MESH.num_layers} layers)", icon='CHECKMARK')
        else:
            box.label(text="Import mesh first", icon="ERROR")
        row = box.row(align=True)
        row.enabled = lmm.is_mesh_loaded()
        row.operator("neurocaptain.setup_mmc", text="Configure", icon="SETTINGS")
        row.operator("neurocaptain.run_mmc",   text="Run",       icon="PLAY")
        # Custom colormap range (MMC)
        box.prop(nc, "viz_custom_range", icon="FCURVE")
        if nc.viz_custom_range:
            row2 = box.row(align=True)
            row2.prop(nc, "viz_vmin", text="Min")
            row2.prop(nc, "viz_vmax", text="Max")

        # ── Redbird ──────────────────────────────────────────────────────
        layout.separator()
        box = layout.box()
        box.label(text="Redbird (Diffusion)", icon="LIGHT_SUN")
        if lmm.is_mesh_loaded():
            box.operator("neurocaptain.show_redbird_config", text="Show Config", icon="INFO")
        else:
            box.label(text="Import mesh first", icon="ERROR")
        box.prop(nc, "redbird_mode")
        row = box.row(align=True)
        row.enabled = lmm.is_mesh_loaded()
        row.operator("neurocaptain.setup_redbird", text="Configure", icon="SETTINGS")
        row.operator("neurocaptain.run_redbird",   text="Run",       icon="PLAY")
        # Custom colormap range (Redbird)
        box.prop(nc, "viz_custom_range", icon="FCURVE")
        if nc.viz_custom_range:
            row2 = box.row(align=True)
            row2.prop(nc, "viz_vmin", text="Min")
            row2.prop(nc, "viz_vmax", text="Max")


# ── Dependencies ─────────────────────────────────────────────────────────────

class NEUROCAPTAIN_PT_dependencies_subpanel(bpy.types.Panel):
    bl_label      = "Dependencies"
    bl_idname     = "NEUROCAPTAIN_PT_dependencies_subpanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category   = "NeuroCaptain"
    bl_parent_id  = "NEUROCAPTAIN_PT_main_panel"
    poll          = _tab_poll("DEPS")

    def draw(self, context):
        layout = self.layout
        from .dependencies import get_missing_dependencies
        missing = get_missing_dependencies()
        box = layout.box()
        if missing:
            box.label(text="Missing Dependencies:", icon="ERROR")
            for dep in missing[:3]:
                box.label(text=f"• {dep}")
            if len(missing) > 3:
                box.label(text=f"• ... and {len(missing) - 3} more")
            from .pkg import (InstallJData, InstallNumPy, InstallSciPy,
                               InstallIso2Mesh, InstallPMMC, InstallRedbird,
                               InstallAllDependencies, CheckDependencies)
            row = box.row()
            row.operator(InstallAllDependencies.bl_idname, text="Install All", icon="IMPORT")
            row.operator(CheckDependencies.bl_idname,      text="Check",       icon="FILE_REFRESH")
            row = box.row()
            row.operator(InstallJData.bl_idname,    text="JData",    icon="FILE_TICK")
            row.operator(InstallNumPy.bl_idname,    text="NumPy",    icon="FILE_TICK")
            row.operator(InstallSciPy.bl_idname,    text="SciPy",    icon="FILE_TICK")
            row = box.row()
            row.operator(InstallIso2Mesh.bl_idname, text="iso2mesh", icon="FILE_TICK")
            row.operator(InstallPMMC.bl_idname,     text="pmmc",     icon="FILE_TICK")
            row.operator(InstallRedbird.bl_idname,  text="redbirdpy", icon="FILE_TICK")
        else:
            from .pkg import CheckDependencies
            box.operator(CheckDependencies.bl_idname,
                         text="All Dependencies Available", icon="CHECKMARK")

        layout.separator()
        layout.label(text="Resources", icon="SHADING_SOLID")
        row = layout.row()
        row.operator("wm.url_open", text="Iso2Mesh", icon="URL").url = "http://iso2mesh.sf.net"
        row.operator("wm.url_open", text="MMC wiki", icon="URL").url = "http://mcx.space/wiki/?Learn#mmc"


# ============================================================================
# REGISTER / UNREGISTER
# ============================================================================

CLASSES = [
    NeuroCaptainSettings,
    NEUROCAPTAIN_OT_import_layered_mesh,
    NEUROCAPTAIN_OT_import_layered_mesh_flexible,
    NEUROCAPTAIN_OT_define_layers,
    NEUROCAPTAIN_OT_setup_mmc,
    NEUROCAPTAIN_OT_run_mmc,
    NEUROCAPTAIN_OT_setup_redbird,
    NEUROCAPTAIN_OT_run_redbird,
    NEUROCAPTAIN_OT_show_mesh_info,
    NEUROCAPTAIN_OT_show_redbird_config,
    NEUROCAPTAIN_OT_add_source,
    NEUROCAPTAIN_OT_add_detector,
    NEUROCAPTAIN_OT_move_optode,
    NEUROCAPTAIN_OT_ensure_optode_constraints,
    NEUROCAPTAIN_PT_main_panel,
    NEUROCAPTAIN_PT_capgen_subpanel,
    NEUROCAPTAIN_PT_optodes_subpanel,
    NEUROCAPTAIN_PT_lightsim_subpanel,
    NEUROCAPTAIN_PT_dependencies_subpanel,
]


def register():
    njloader.register()
    schematic_2d.register()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.neurocaptain_settings = bpy.props.PointerProperty(type=NeuroCaptainSettings)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.neurocaptain_settings
    schematic_2d.unregister()
    njloader.unregister()