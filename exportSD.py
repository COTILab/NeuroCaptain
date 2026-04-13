import bpy
import numpy as np
from scipy.io import savemat
from scipy.spatial.distance import pdist, squareform
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from datetime import datetime
import json

# EXPORT FUNCTION

def export_sd_all_landmarks(filepath):
    """Export with all landmarks as anchors"""
    
    # Get optodes
    sources = [obj for obj in bpy.data.objects if obj.name.startswith("Source_")]
    detectors = [obj for obj in bpy.data.objects if obj.name.startswith("Detector_")]
    sources.sort(key=lambda x: x.name)
    detectors.sort(key=lambda x: x.name)
    
    all_optodes = sources + detectors
    optode_positions_3d = np.array([obj.location for obj in all_optodes])
    
    # Get ALL landmarks
    if 'LandmarkMesh' not in bpy.data.objects:
        print("ERROR: LandmarkMesh not found")
        return
    
    lm = bpy.data.objects['LandmarkMesh']
    labels = ["Nz", "Iz", "Lpa", "Rpa", "Cz", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "FT7", "F7", "AF7", "Fp1", "TP7", "P7", "PO7", "O1", "FT8", "F8", "AF8", "Fp2", "TP8", "P8", "PO8", "O2", "FC1", "FC3", "FC5", "FC2", "FC4", "FC6", "F1", "F3", "F5", "F2", "F4", "F6", "AF3", "AF4", "CP1", "CP3", "CP5", "CP2", "CP4", "CP6", "P1", "P3", "P5", "P2", "P4", "P6", "PO3", "PO4", "FT9", "F9", "", "", "TP9", "P9", "PO9", "O9", "FT10", "F10", "", "", "TP10", "P10", "PO10", "O10"]
    
    # Create dummies at ALL landmark positions
    dummy_positions_3d = []
    dummy_labels = []
    
    for i, v in enumerate(lm.data.vertices):
        if i < len(labels) and labels[i]:
            label = labels[i]
            pos = lm.matrix_world @ v.co
            dummy_positions_3d.append([pos.x, pos.y, pos.z])
            dummy_labels.append(label)
    
    dummy_positions_3d = np.array(dummy_positions_3d)
    
    print(f"\n1. Using {len(dummy_positions_3d)} landmarks as anchors")
    
    # Create anchors (ALL dummies)
    anchors = []
    for i, label in enumerate(dummy_labels):
        anchor_idx = len(all_optodes) + i + 1
        anchors.append({'index': anchor_idx, 'label': label})
    
    # Combine for MDS
    all_positions_3d = np.vstack([optode_positions_3d, dummy_positions_3d])
    
    print(f"2. MDS projection ({len(all_positions_3d)} total positions)...")
    
    # Classical MDS
    distances_3d = squareform(pdist(all_positions_3d))
    
    n = len(all_positions_3d)
    D_squared = distances_3d ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_squared @ H
    
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    lambda_sqrt = np.sqrt(np.maximum(eigenvalues[:2], 0))
    positions_2d_xy = eigenvectors[:, :2] * lambda_sqrt
    
    # Verify
    distances_2d = squareform(pdist(positions_2d_xy))
    dist_error = np.mean(np.abs(distances_2d - distances_3d))
    correlation = np.corrcoef(distances_2d.flatten(), distances_3d.flatten())[0, 1]
    
    print(f"  Distance error: {dist_error:.1f}mm, r={correlation:.3f}")
    
    # Add Z column
    positions_2d = np.column_stack([positions_2d_xy, np.zeros(len(positions_2d_xy))])
    
    sources_2d = positions_2d[:len(sources)]
    detectors_2d = positions_2d[len(sources):len(all_optodes)]
    dummies_2d = positions_2d[len(all_optodes):]
    
    # Get springs
    springs = []
    if "spring_pairs" in bpy.context.scene:
        springs = [[p[0] + 1, p[1] + 1] for p in bpy.context.scene["spring_pairs"]]
    
    print(f"\n3. Springs: {len(springs)}")
    
    # Create anchor list
    anchor_list = np.empty((len(anchors), 2), dtype=object)
    for i, anchor in enumerate(anchors):
        anchor_list[i, 0] = np.array([[anchor['index']]], dtype=np.float64)
        label_str = anchor['label']
        anchor_list[i, 1] = np.array([label_str], dtype=f'<U{len(label_str)}')
    
    # Export
    sd_struct = {
        'SrcPos': sources_2d,
        'DetPos': detectors_2d,
        'DummyPos': dummies_2d,
        'nSrcs': np.array([[len(sources_2d)]], dtype=np.float64),
        'nDets': np.array([[len(detectors_2d)]], dtype=np.float64),
        'AnchorList': anchor_list,
        'SpringList': np.array(springs) if springs else np.empty((0, 2))
    }
    
    savemat(filepath, {'SD': sd_struct}, format='5', oned_as='column')
    print(f"4. Exported to: {filepath}")


# WORKFLOW

print("\n" + "="*70)
print("EXPORT WITH ALL LANDMARKS (~77 ANCHORS)")
print("="*70)

# Save originals
print("\n1. Saving originals...")
original_positions = {'sources': {}, 'detectors': {}}
for obj in bpy.data.objects:
    if obj.name.startswith("Source_"):
        original_positions['sources'][obj.name] = {'x': float(obj.location.x), 'y': float(obj.location.y), 'z': float(obj.location.z)}
    elif obj.name.startswith("Detector_"):
        original_positions['detectors'][obj.name] = {'x': float(obj.location.x), 'y': float(obj.location.y), 'z': float(obj.location.z)}

with open(r"C:\Users\Ashlyn\Desktop\original_alllandmarks.json", 'w') as f:
    json.dump(original_positions, f, indent=2)
print("   ✓ Saved")

# Create springs
print("\n2. Creating springs...")
sources = [obj for obj in bpy.data.objects if obj.name.startswith("Source_")]
detectors = [obj for obj in bpy.data.objects if obj.name.startswith("Detector_")]
all_opts = sources + detectors
positions = np.array([obj.location for obj in all_opts])
distances = squareform(pdist(positions))

springs = []
for i in range(len(all_opts)):
    dists_idx = [(distances[i, j], j) for j in range(len(all_opts)) if j != i]
    dists_idx.sort()
    for dist, j in dists_idx[:6]:
        if dist < 60:
            pair = tuple(sorted([i, j]))
            if pair not in [tuple(sorted(s)) for s in springs]:
                springs.append([i, j])

bpy.context.scene["spring_pairs"] = springs
print(f"   ✓ {len(springs)} springs")

# Export
print("\n3. Exporting with ALL landmarks...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
export_path = rf"C:\Users\Ashlyn\Desktop\probe_ALLLANDMARKS_{timestamp}.SD"

export_sd_all_landmarks(export_path)

print(f"\n{'='*70}")
print(f"✓ EXPORTED: {export_path}")
print(f"{'='*70}")

import subprocess
subprocess.Popen(f'explorer /select,"{export_path}"')

print("\nDELETE AND IMPORT:")
print("for obj in list(bpy.data.objects):")
print("    if obj.name.startswith('Source_') or obj.name.startswith('Detector_'):")
print("        bpy.data.objects.remove(obj, do_unlink=True)")
print(f"bpy.ops.neurocaptain.import_sd_probe(filepath=r'{export_path}')")
print("\nThen run 'Relative Layout Preservation Check' artifact")