@echo off
set BLENDER="C:\Users\Ashlyn\Downloads\blender-3.6.23-windows-x64\blender-3.6.23-windows-x64\blender.exe"
set SCRIPT="C:\Users\Ashlyn\OneDrive\Documents\GitHub\BrainCaptain\tests\repro_cap_boolean_issue.py"

echo Running WITHOUT --override-boolean (matches current CI) on Blender 3.6.23...
%BLENDER% --background --factory-startup --python-exit-code 1 --python %SCRIPT%

echo.
echo ============================================================
echo Running WITH --override-boolean on Blender 3.6.23...
echo ============================================================
echo.
%BLENDER% --background --factory-startup --python-exit-code 1 --python %SCRIPT% -- --override-boolean

pause
