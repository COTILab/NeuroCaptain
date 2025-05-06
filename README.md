# NeuroCaptain

- **Authors**: Ashlyn McCann (mccann.as@northeastern.edu) and Qianqian Fang (q.fang@neu.edu)
- **License**: GNU General Public License Version 3 (GPLv3)
- **Version**: v2025 (v0.1)
- **Website**: [neurojson.org/NeuroCaptain](http://neurojson.org/NeuroCaptain)
- **Acknowledgment**: This project is supported by NIH awards [R01-EB026998]() and [U24-NS124027](https://reporter.nih.gov/search/dXkcyoaEQkaRrkpQoOnEBw/project-details/10308329).

---

## Introduction

**NeuroCaptain** is a Blender add-on designed to create 3D-printable head caps from 3D head models for neuroimaging applications. By integrating **Iso2Mesh**, **Brain2Mesh**, and Blender’s native tools, NeuroCaptain streamlines the head cap design process while allowing extensive customization. Iso2Mesh and Brain2Mesh run via GNU Octave or MATLAB, interfacing with Blender through the `oct2py` module and Blender’s `bpy` API.

Users can select a head model from the built-in atlas library or import a custom model. NeuroCaptain supports three methods for defining landmarks:

- **Automatic Calculation** of standard 10-20, 10-10, or 10-5 system landmarks.
- **Custom Layout Generation** by manually selecting vertices.
- **Custom Geometry Labeling** by importing a user-defined geometry.

The landmark geometry is integrated into the head surface using Blender’s procedural modeling tool, **Geometry Nodes**.

Once landmarks are integrated, a wireframe head cap is generated through edge extraction and thickness adjustment — all achievable with a single click — resulting in a ready-to-print 3D model.

If you use NeuroCaptain in your research, please cite:

> Ashlyn McCann, Edward Xu, Fan-Yu Yen, Noah Joseph, and Qianqian Fang,  
> *"Creating anatomically derived, standardized, customizable, and three-dimensional printable head caps for functional neuroimaging,"*  
> Neurophotonics, 12(1), 015016 (2025).  
> [https://doi.org/10.1117/1.NPh.12.1.015016](https://doi.org/10.1117/1.NPh.12.1.015016)

---

## Installation

### Prerequisites

- MATLAB or GNU Octave installed and added to your system's environment variables.

### Prepackaged NeuroCaptain-Blender Integration

- **Windows**: [Download Link]
- **Linux**: [Download Link]

### Manual Installation

1. Download the NeuroCaptain source code as a ZIP archive: [GitHub Repository Link].
2. Download and install dependencies: **Iso2Mesh** and **Brain2Mesh**.
3. Add Iso2Mesh and Brain2Mesh to your MATLAB/Octave path.
4. Install required Python modules:
   - `oct2py`
   - `jdata`
   - `bjdata`
   
   Steps:
   - In Blender’s Scripting tab, run:
     ```python
     bpy.utils.user_resource('SCRIPTS', path='modules')
     ```
   - Use the output path to install modules:
     ```bash
     pip install MODULE_NAME --target="OUTPUT_PATH_FROM_ABOVE"
     ```

5. Open Blender (version 3.4 or later).
6. Navigate to `Edit > Preferences > Add-ons > Install`.
7. Select the downloaded `NeuroCaptain.zip` file to install the add-on.

---
