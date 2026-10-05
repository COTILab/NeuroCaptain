@echo off
set BLENDER="C:\Users\Ashlyn\OneDrive - Northeastern University\COTI\fNIRS Cap\Blender3.6.9_NeuroCaptain\blender.exe"
set SCRIPT="C:\Users\Ashlyn\OneDrive\Documents\GitHub\BrainCaptain\tests\repro_cap_boolean_issue.py"

echo Running WITHOUT --override-boolean (matches current CI)...
%BLENDER% --background --factory-startup --python-exit-code 1 --python %SCRIPT%

echo.
echo ============================================================
echo Running WITH --override-boolean...
echo ============================================================
echo.
%BLENDER% --background --factory-startup --python-exit-code 1 --python %SCRIPT% -- --override-boolean

pause
