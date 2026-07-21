import bpy
import bmesh
from mathutils import Vector
from mathutils.bvhtree import BVHTree
import random


# ========== PROPERTY GROUP ==========

class OptodeModule(bpy.types.PropertyGroup):
    """Property group to store optode module data"""
    name: bpy.props.StringProperty(
        name="Module Name",
        default="Module"
    )
    optode_names: bpy.props.StringProperty(
        name="Optode Names",
        description="Comma-separated list of optode names in this module",
        default=""
    )
    is_locked: bpy.props.BoolProperty(
        name="Locked",
        description="Whether this module is locked for group movement",
        default=True
    )
    color: bpy.props.FloatVectorProperty(
        name="Module Color",
        subtype='COLOR',
        default=(0.8, 0.3, 0.3),
        min=0.0,
        max=1.0
    )


# ========== OPERATORS ==========

class NEUROCAPTAIN_OT_create_optode_module(bpy.types.Operator):
    """Create a module from selected optodes"""
    bl_idname = "neurocaptain.create_optode_module"
    bl_label = "Create Optode Module"
    bl_options = {'REGISTER', 'UNDO'}
    
    module_name: bpy.props.StringProperty(
        name="Module Name",
        description="Name for this optode group",
        default="Module"
    )
    
    def invoke(self, context, event):
        num_modules = len(context.scene.neurocaptain_optode_modules)
        self.module_name = f"Module_{num_modules + 1}"
        return context.window_manager.invoke_props_dialog(self)
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "module_name")
    
    def execute(self, context):
        selected_optodes = [obj for obj in context.selected_objects 
                           if obj.name.startswith(('Source_', 'Detector_'))]
        
        if len(selected_optodes) < 2:
            self.report({'ERROR'}, "Select at least 2 optodes to create a module")
            return {'CANCELLED'}
        
        module = context.scene.neurocaptain_optode_modules.add()
        module.name = self.module_name
        module.optode_names = ",".join([opt.name for opt in selected_optodes])
        module.is_locked = True
        module.color = (random.random(), random.random(), random.random())
        
        for opt in selected_optodes:
            opt["module_name"] = self.module_name
        
        self.report({'INFO'}, f"Created module '{self.module_name}' with {len(selected_optodes)} optodes")
        return {'FINISHED'}


class NEUROCAPTAIN_OT_delete_optode_module(bpy.types.Operator):
    """Delete the module currently selected in the list above"""
    bl_idname = "neurocaptain.delete_optode_module"
    bl_label = "Delete Module"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        modules = context.scene.neurocaptain_optode_modules
        index = context.scene.neurocaptain_optode_modules_index
        
        if 0 <= index < len(modules):
            module = modules[index]
            
            optode_names = module.optode_names.split(",")
            for name in optode_names:
                if name in bpy.data.objects:
                    opt = bpy.data.objects[name]
                    if "module_name" in opt:
                        del opt["module_name"]
            
            modules.remove(index)
            context.scene.neurocaptain_optode_modules_index = max(0, index - 1)
            
            self.report({'INFO'}, "Deleted module")
        
        return {'FINISHED'}


class NEUROCAPTAIN_OT_select_module_optodes(bpy.types.Operator):
    """Select all optodes in this module"""
    bl_idname = "neurocaptain.select_module_optodes"
    bl_label = "Select Module Optodes"
    bl_options = {'REGISTER', 'UNDO'}
    
    module_index: bpy.props.IntProperty()
    
    def execute(self, context):
        modules = context.scene.neurocaptain_optode_modules
        
        if 0 <= self.module_index < len(modules):
            module = modules[self.module_index]
            optode_names = module.optode_names.split(",")
            
            bpy.ops.object.select_all(action='DESELECT')
            
            for name in optode_names:
                if name in bpy.data.objects:
                    bpy.data.objects[name].select_set(True)
            
            self.report({'INFO'}, f"Selected {len(optode_names)} optodes from '{module.name}'")
        
        return {'FINISHED'}


class NEUROCAPTAIN_OT_toggle_module_lock(bpy.types.Operator):
    """Toggle module lock status"""
    bl_idname = "neurocaptain.toggle_module_lock"
    bl_label = "Toggle Lock"
    bl_options = {'REGISTER', 'UNDO'}
    
    module_index: bpy.props.IntProperty()
    
    def execute(self, context):
        modules = context.scene.neurocaptain_optode_modules
        
        if 0 <= self.module_index < len(modules):
            module = modules[self.module_index]
            module.is_locked = not module.is_locked
            
            status = "locked" if module.is_locked else "unlocked"
            self.report({'INFO'}, f"Module '{module.name}' {status}")
        
        return {'FINISHED'}


class NEUROCAPTAIN_OT_move_module_group(bpy.types.Operator):
    """Move all optodes in a module as a rigid group"""
    bl_idname = "neurocaptain.move_module_group"
    bl_label = "Move Module as Group"
    bl_options = {'REGISTER', 'UNDO', 'GRAB_CURSOR', 'BLOCKING'}
    
    module_index: bpy.props.IntProperty()

    initial_positions = {}
    initial_rotations = {}
    optodes = []
    bvh_tree = None
    original_distances = {}
    _initial_mouse_region = None
    _current_offset = Vector((0, 0, 0))

    def modal(self, context, event):
        context.area.tag_redraw()

        if event.type == 'MOUSEMOVE':
            region = context.region
            rv3d = context.region_data

            right_vector = rv3d.view_rotation @ Vector((1, 0, 0))
            up_vector = rv3d.view_rotation @ Vector((0, 1, 0))

            delta_x = event.mouse_region_x - self._initial_mouse_region[0]
            delta_y = event.mouse_region_y - self._initial_mouse_region[1]

            scale = 0.01
            self._current_offset = (right_vector * delta_x + up_vector * delta_y) * scale

            if self.apply_movement(context):
                context.area.header_text_set(f"Move Module: {self._current_offset.length:.2f}")
            else:
                context.area.header_text_set("⚠ Movement limited by constraints")

            return {'RUNNING_MODAL'}

        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            context.area.header_text_set(None)
            context.view_layer.update()
            return {'FINISHED'}

        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            for opt in self.optodes:
                opt.location = self.initial_positions[opt.name]
                opt.rotation_euler = self.initial_rotations[opt.name]
            context.view_layer.update()
            context.area.header_text_set(None)
            return {'CANCELLED'}
        
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'}:
            return {'PASS_THROUGH'}
        
        return {'RUNNING_MODAL'}
    
    def invoke(self, context, event):
        modules = context.scene.neurocaptain_optode_modules
        
        if not (0 <= self.module_index < len(modules)):
            self.report({'ERROR'}, "Invalid module")
            return {'CANCELLED'}
        
        module = modules[self.module_index]
        
        if not module.is_locked:
            self.report({'WARNING'}, "Module is unlocked - lock it first for group movement")
            return {'CANCELLED'}
        
        optode_names = module.optode_names.split(",")
        self.optodes = [bpy.data.objects[name] for name in optode_names if name in bpy.data.objects]
        
        if len(self.optodes) < 2:
            self.report({'ERROR'}, "Module has too few optodes")
            return {'CANCELLED'}
        
        # Find head mesh
        head_mesh = bpy.data.objects.get("headmesh")
        if not head_mesh:
            for opt in self.optodes:
                for con in opt.constraints:
                    if con.type == 'SHRINKWRAP' and con.target:
                        head_mesh = con.target
                        break
                if head_mesh:
                    break
        
        if not head_mesh:
            self.report({'ERROR'}, "No head mesh found")
            return {'CANCELLED'}
        
        # Build BVH tree
        depsgraph = context.evaluated_depsgraph_get()
        eval_obj = head_mesh.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(head_mesh.matrix_world)
        self.bvh_tree = BVHTree.FromBMesh(bm)
        bm.free()
        eval_obj.to_mesh_clear()
        
        # Store initial state
        for opt in self.optodes:
            self.initial_positions[opt.name] = opt.location.copy()
            self.initial_rotations[opt.name] = opt.rotation_euler.copy()
        
        # Store distances
        for i, opt1 in enumerate(self.optodes):
            for opt2 in self.optodes[i+1:]:
                key = tuple(sorted([opt1.name, opt2.name]))
                self.original_distances[key] = (opt1.location - opt2.location).length
        
        self._initial_mouse_region = (event.mouse_region_x, event.mouse_region_y)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def apply_movement(self, context):
        """Apply movement maintaining distances"""
        new_positions = {}
        new_rotations = {}

        for opt in self.optodes:
            new_pos = self.initial_positions[opt.name] + self._current_offset

            # Project to surface
            location, normal, _, _ = self.bvh_tree.find_nearest(new_pos)
            # Only reject points whose normals face nearly straight down (chin
            # underside).  -0.8 ≈ 144° from up — much less restrictive than -0.3.
            if not location or normal.z < -0.8:
                return False

            new_positions[opt.name] = location
            new_rotations[opt.name] = Vector((0, 0, 1)).rotation_difference(normal).to_euler()

        # Validate distances — 15% tolerance gives room for surface curvature
        # without hard-stopping on every hill/valley.
        tolerance = 0.15
        for i, opt1 in enumerate(self.optodes):
            for opt2 in self.optodes[i+1:]:
                key = tuple(sorted([opt1.name, opt2.name]))
                orig_dist = self.original_distances[key]
                new_dist = (new_positions[opt1.name] - new_positions[opt2.name]).length
                ratio = new_dist / orig_dist
                if ratio < (1 - tolerance) or ratio > (1 + tolerance):
                    return False
        
        # Apply positions (hooks will automatically update connection mesh)
        for opt in self.optodes:
            opt.location = new_positions[opt.name]
            opt.rotation_euler = new_rotations[opt.name]
        
        context.view_layer.update()
        return True


# ========== UI LIST ==========

class NEUROCAPTAIN_UL_optode_modules(bpy.types.UIList):
    """UI List for optode modules"""
    
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        module = item
        
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            
            lock_icon = 'LOCKED' if module.is_locked else 'UNLOCKED'
            lock_op = row.operator("neurocaptain.toggle_module_lock", 
                                  text="", icon=lock_icon, emboss=False)
            lock_op.module_index = index
            
            row.prop(module, "name", text="", emboss=False, icon='GROUP')
            
            num_optodes = len(module.optode_names.split(",")) if module.optode_names else 0
            row.label(text=f"({num_optodes})")
            
            select_op = row.operator("neurocaptain.select_module_optodes", 
                                    text="", icon='RESTRICT_SELECT_OFF', emboss=False)
            select_op.module_index = index
            
        elif self.layout_type == 'GRID':
            layout.alignment = 'CENTER'
            layout.prop(module, "name", text="", emboss=False)


# ========== REGISTRATION ==========

def register():
    bpy.utils.register_class(OptodeModule)
    bpy.utils.register_class(NEUROCAPTAIN_UL_optode_modules)
    bpy.utils.register_class(NEUROCAPTAIN_OT_create_optode_module)
    bpy.utils.register_class(NEUROCAPTAIN_OT_delete_optode_module)
    bpy.utils.register_class(NEUROCAPTAIN_OT_select_module_optodes)
    bpy.utils.register_class(NEUROCAPTAIN_OT_toggle_module_lock)
    bpy.utils.register_class(NEUROCAPTAIN_OT_move_module_group)
    
    bpy.types.Scene.neurocaptain_optode_modules = bpy.props.CollectionProperty(
        type=OptodeModule
    )
    bpy.types.Scene.neurocaptain_optode_modules_index = bpy.props.IntProperty(
        name="Active Module Index",
        default=0
    )


def unregister():
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_move_module_group)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_toggle_module_lock)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_select_module_optodes)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_delete_optode_module)
    bpy.utils.unregister_class(NEUROCAPTAIN_OT_create_optode_module)
    bpy.utils.unregister_class(NEUROCAPTAIN_UL_optode_modules)
    bpy.utils.unregister_class(OptodeModule)
    
    del bpy.types.Scene.neurocaptain_optode_modules
    del bpy.types.Scene.neurocaptain_optode_modules_index
