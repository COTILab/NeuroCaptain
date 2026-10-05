@echo off
set BLENDER="C:\Users\Ashlyn\OneDrive - Northeastern University\COTI\fNIRS Cap\Blender3.6.9_NeuroCaptain\blender.exe"
set REPO_ROOT=C:\Users\Ashlyn\OneDrive\Documents\GitHub\BrainCaptain

%BLENDER% --background --factory-startup --python-exit-code 1 --python "%REPO_ROOT%\tests\run_tests.py"

pause
