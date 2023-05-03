import bpy
from bpy import context
from .utils import *
import oct2py
import bmesh
import numpy as np
import jdata as jd
import pathlib
from bpy.types import Operator
import os
import pathlib

#mode = bpy.context.active_object.mode
#bpy.ops.object.mode_set(mode = 'OBJECT')

class init_points(Operator):
    bl_label = 'Select vertices to calculate 10-20 points'
    bl_description = "Click this button to "
    bl_idname = 'braincapgen.init_points'
   # bl_options = {"REGISTER", "UNDO"}

    def func(self):
        outputdir = GetBPWorkFolder();
        obj = bpy.context.view_layer.objects.active
        #take selected points and save to file
        bpy.ops.object.mode_set(mode='OBJECT')
        selectedverts = [v for v in bpy.context.active_object.data.vertices if v.select]
        
        vselect = []
        for n in range(len(selectedverts)):
            vert = selectedverts[n].co
            v_global = obj.matrix_world @ vert
            vselect.append(v_global)
        vs = np.array(vselect)
        print("verts = ", vs)
        #export the head surface
        
        verts = []
        for n in range(len(obj.data.vertices)):
            vert = obj.data.vertices[n].co
            v_global = obj.matrix_world @ vert
            verts.append(v_global)
        faces = [(np.array(face.vertices[:])+1).tolist() for face in obj.data.polygons]
        v = np.array(verts)
        f = np.array(faces)

        meshdata={'_DataInfo_': {'JMeshVersion': '0.5', 'Comment':'Created by BlenderPhotonics (http:\/\/mcx.space\/BlenderPhotonics)'},
            'MeshVertex3':v, 'MeshTri3':f, 'param': {'initpoints':vs, 'p1':10, 'p2':20}}
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
        AddMeshFromNodeFace(outputmesh['MeshVertex3'],outputmesh['MeshTri3'],'LandmarkMesh');
        bpy.context.view_layer.objects.active=bpy.data.objects['LandmarkMesh']

        ShowMessageBox("Generating 10-20 points is completed", "BrainCapGen")

    def execute(self, context):
        print("begin to calculate 10-20 landmarks")
        self.func()
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

def register():
        bpy.utils.register_class(init_points)


def unregister():
        bpy.utils.unregister_class(init_points)


if __name__ == "__main__":
        register()


# add object to scene 

#oct2py.feval 

#import the brain mesh from matlab and put in scene 
