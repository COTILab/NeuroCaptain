import bpy
from bpy import context
import subprocess
import sys
from .utils import *
import oct2py
import numpy as np
import jdata as jd
import pathlib
from bpy.types import Operator, PropertyGroup
import os
from bpy.props import EnumProperty, StringProperty, CollectionProperty

#mode = bpy.context.active_object.mode
#bpy.ops.object.mode_set(mode = 'OBJECT')

class brain1020mesh(Operator):
    bl_label = 'Select vertices to calculate 10-20 points'
    bl_description = "Click this button to "
    bl_idname = 'braincapgen.init_points'
    action: EnumProperty(
        items=[
            ('NZ_SELECT', 'nz_select', 'nz_select'),
            ('LPA_SELECT', 'lpa_select', 'lpa_select'),
            ('RPA_SELECT', 'RPA_select', 'rpa_select'),
            ('BRAIN1020_MESH', 'brain1020_mesh', 'brain1020_mesh')
        ]
    )
    point1 : bpy.props.FloatProperty(name = "p1", default = 10)
    point2 : bpy.props.FloatProperty(name = "p2", default = 10)
   # bl_options = {"REGISTER", "UNDO"}

    def execute(self,context):
        outputdir = GetBPWorkFolder();
        obj = bpy.context.view_layer.objects.active
        #take selected points and save to file
        bpy.ops.object.mode_set(mode='OBJECT')

        if self.action == 'NZ_SELECT':
            self.nz_select(context=context)

        elif self.action == 'LPA_SELECT':
            self.lpa_select(context=context)
            
        elif self.action == 'RPA_SELECT':
            self.rpa_select(context=context)

        elif self.action == 'BRAIN1020_MESH':
            #point1 : bpy.props.FloatProperty(name = "p1", default = 10)
            #point2 : bpy.props.FloatProperty(name = "p2", default = 10)
            global p1, p2
            #global p2
            p1 = self.point1
            p2 = self.point2
            self.brain1020_mesh(context=context)

        return {"FINISHED"}
            
       

    #def execute(self, context):
    #    print("begin to calculate 10-20 landmarks")
     #   self.func()
    #    return {"FINISHED"}

    def invoke(self, context, event):
        #if self.action == 'BRAIN1020_MESH':
        return context.window_manager.invoke_props_dialog(self)

    @staticmethod
    def nz_select(context):
        bpy.ops.object.mode_set(mode='OBJECT')
        obj = bpy.context.view_layer.objects.active
        #vert = obj.data.vertices[n].co
        #take selected points and save to file
        bpy.ops.object.mode_set(mode='OBJECT')
        selectedverts_nz = [v for v in bpy.context.active_object.data.vertices if v.select]
        
        vselect_nz = []
        for n in range(len(selectedverts_nz)):
            vert_nz = selectedverts_nz[n].co
            v_global_nz = obj.matrix_world @ vert_nz
            vselect_nz.append(v_global_nz)
        global nz 
        nz = np.array(vselect_nz)
        print(nz)
        pass

    @staticmethod
    def lpa_select(context):
        bpy.ops.object.mode_set(mode='OBJECT')
        obj = bpy.context.view_layer.objects.active
        #vert = obj.data.vertices[n].co
        #take selected points and save to file
        bpy.ops.object.mode_set(mode='OBJECT')
        selectedverts_lpa = [v for v in bpy.context.active_object.data.vertices if v.select]
        
        vselect_lpa = []
        for n in range(len(selectedverts_lpa)):
            vert_lpa = selectedverts_lpa[n].co
            v_global_lpa = obj.matrix_world @ vert_lpa
            vselect_lpa.append(v_global_lpa)
        global lpa 
        lpa= np.array(vselect_lpa)
        pass 

    @staticmethod
    def rpa_select(context):
        bpy.ops.object.mode_set(mode='OBJECT')
        obj = bpy.context.view_layer.objects.active
        #vert = obj.data.vertices[n].co
        #take selected points and save to file
        bpy.ops.object.mode_set(mode='OBJECT')
        selectedverts_rpa = [v for v in bpy.context.active_object.data.vertices if v.select]
        
        vselect_rpa = []
        for n in range(len(selectedverts_rpa)):
            vert_rpa = selectedverts_rpa[n].co
            v_global_rpa = obj.matrix_world @ vert_rpa
            vselect_rpa.append(v_global_rpa)
        global rpa    
        rpa= np.array(vselect_rpa)

        pass

    @staticmethod
    def brain1020_mesh(context):
        outputdir = GetBPWorkFolder();
        obj = bpy.context.view_layer.objects.active
        vs = np.vstack((nz, lpa, rpa))
        verts = []
        for n in range(len(obj.data.vertices)):
            vert = obj.data.vertices[n].co
            v_global = obj.matrix_world @ vert
            verts.append(v_global)
        faces = [(np.array(face.vertices[:])+1).tolist() for face in obj.data.polygons]
        v = np.array(verts)
        f = np.array(faces)

        meshdata={'_DataInfo_': {'JMeshVersion': '0.5', 'Comment':'Created by BlenderPhotonics (http:\/\/mcx.space\/BlenderPhotonics)'},
            'MeshVertex3':v, 'MeshTri3':f, 'param': {'initpoints':vs, 'p1':p1, 'p2':p2}}
        jd.save(meshdata,os.path.join(outputdir,'brain1020input.jmsh'))

            #Use these files as inputs for a brain mesh generation script in Matlab
        try:
            if(bpy.context.scene.blender_photonics.backend == "octave"):
                import oct2py as op
                oc = op.Oct2Py()
            else:
                import matlab.engine as op
                oc = op.start_matlab()
        except ImportError:
            raise ImportError('To run this feature, you must install the `oct2py` or `matlab.engine` Python module first, based on your choice of the backend')
        oc.addpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'script'))
        oc.feval('brain1020mesh',os.path.join(outputdir,'brain1020input.jmsh'))

            #load the resulting file
        outputmesh=jd.load(os.path.join(outputdir,'brain1020output.jmsh'))
        AddMeshFromNodeFace(outputmesh['MeshVertex3'],(np.array(outputmesh['MeshTri3'])-1).tolist(),'LandmarkMesh');
        bpy.context.view_layer.objects.active=bpy.data.objects['LandmarkMesh']

        ShowMessageBox("Generating 10-20 points is completed", "BrainCapGen")

        pass
        
        
        
    

def register():
        bpy.utils.register_class(init_points)


def unregister():
        bpy.utils.unregister_class(init_points)


if __name__ == "__main__":
        register()

        bpy.ops.braincapgen.brain1020mesh('INVOKE_DEFAULT')

# add object to scene 

#oct2py.feval 

#import the brain mesh from matlab and put in scene 
