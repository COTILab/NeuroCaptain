import bpy
import subprocess
import sys
import os
import pathlib
from .dependencies import check_dependencies, show_error_message


def _get_addon_dir():
    """Return the modules directory next to the addon, creating it if needed."""
    addon_dir = os.path.join(
        os.path.abspath(pathlib.Path(__file__).resolve().parent.parent),
        "modules",
    )
    if not os.path.exists(addon_dir):
        os.makedirs(addon_dir)
    return addon_dir


def _pip_install(package, addon_dir):
    """Run pip install for a single package into addon_dir. Returns (success, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", package, "--target=" + addon_dir],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


def _make_installer(bl_idname, bl_label, bl_description, packages, prereqs=None):
    """Create a Blender operator class that pip-installs the given packages.

    prereqs: list of (condition_callable, package_name) tuples.
             condition_callable() returns True if the prereq should be installed.
    """

    def execute(self, context):
        try:
            addon_dir = _get_addon_dir()

            # Install prerequisites
            if prereqs:
                for condition_fn, prereq_pkg in prereqs:
                    if condition_fn():
                        ok, stderr = _pip_install(prereq_pkg, addon_dir)
                        if not ok:
                            show_error_message(
                                f"Failed to install {prereq_pkg} (prerequisite): {stderr}",
                                "Installation Failed",
                            )
                            return {"FINISHED"}

            # Install main packages
            for pkg in packages:
                ok, stderr = _pip_install(pkg, addon_dir)
                if not ok:
                    show_error_message(
                        f"Failed to install {pkg}: {stderr}",
                        "Installation Failed",
                    )
                    return {"FINISHED"}

            check_dependencies()

            # Build success message listing all installed packages
            all_pkgs = []
            if prereqs:
                for condition_fn, prereq_pkg in prereqs:
                    if condition_fn():
                        all_pkgs.append(prereq_pkg)
            all_pkgs.extend(packages)
            show_error_message(
                f"{' and '.join(all_pkgs)} installed successfully! ",
                "Installation Complete",
            )

        except Exception as e:
            show_error_message(
                f"Error installing {packages[0]}: {str(e)}", "Installation Error"
            )

        return {"FINISHED"}

    cls = type(
        f"Install_{'_'.join(packages)}",  # class name (internal)
        (bpy.types.Operator,),
        {
            "bl_idname": bl_idname,
            "bl_label": bl_label,
            "bl_description": bl_description,
            "bl_options": {"REGISTER", "UNDO"},
            "execute": execute,
        },
    )
    return cls


# ---------------------------------------------------------------------------
# Condition helpers for prerequisites
# ---------------------------------------------------------------------------

def _is_windows():
    import platform
    return platform.system() == "Windows"


def _always():
    return True


# ---------------------------------------------------------------------------
# Installer classes created via factory
# ---------------------------------------------------------------------------

InstallJData = _make_installer(
    "blenderphotonics.install_jdata",
    "Install JData",
    "Install JData package for JSON/JMesh operations",
    ["jdata"],
)

InstallNumPy = _make_installer(
    "blenderphotonics.install_numpy",
    "Install NumPy",
    "Install NumPy package for numerical operations",
    ["numpy"],
)

InstallSciPy = _make_installer(
    "blenderphotonics.install_scipy",
    "Install SciPy",
    "Install SciPy package for scientific computing operations",
    ["scipy"],
)

InstallIso2Mesh = _make_installer(
    "blenderphotonics.install_iso2mesh",
    "Install iso2mesh",
    "Install iso2mesh package for mesh generation operations",
    ["iso2mesh"],
    prereqs=[(_always, "scipy")],
)

InstallPMCX = _make_installer(
    "blenderphotonics.install_pmcx",
    "Install pmcx",
    "Install pmcx package for Monte Carlo eXtreme simulations",
    ["pmcx"],
)

InstallPMMC = _make_installer(
    "blenderphotonics.install_pmmc",
    "Install pmmc",
    "Install pmmc package for Mesh-based Monte Carlo simulations",
    ["pmmc"],
    prereqs=[(_is_windows, "sparse_numba")],
)

InstallRedbird = _make_installer(
    "blenderphotonics.install_redbirdpy",
    "Install redbirdpy",
    "Install redbirdpy package for FEM diffuse optical forward simulations",
    ["redbirdpy"],
)


# ---------------------------------------------------------------------------
# InstallAllDependencies -- kept as an explicit class (partial-failure report)
# ---------------------------------------------------------------------------

class InstallAllDependencies(bpy.types.Operator):
    bl_idname = "blenderphotonics.install_all_deps"
    bl_label = "Install All Dependencies"
    bl_description = "Install all required Python packages for BlenderPhotonics"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        try:
            addon_dir = _get_addon_dir()

            import platform

            packages = ["jdata", "numpy", "scipy", "iso2mesh", "pmcx"]

            # Add Windows-specific dependency for pmmc
            if platform.system() == "Windows":
                packages.append("sparse_numba")

            packages.append("pmmc")
            packages.append("redbirdpy")

            failed_packages = []

            for package in packages:
                ok, _stderr = _pip_install(package, addon_dir)
                if not ok:
                    failed_packages.append(package)

            # Update dependency status
            check_dependencies()

            if failed_packages:
                show_error_message(
                    f"Some packages failed to install: {', '.join(failed_packages)}. Please try installing them individually.",
                    "Partial Installation",
                )
            else:
                show_error_message(
                    "All dependencies installed successfully! ",
                    "Installation Complete",
                )

        except Exception as e:
            show_error_message(
                f"Error installing dependencies: {str(e)}", "Installation Error"
            )

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# CheckDependencies -- left as-is
# ---------------------------------------------------------------------------

class CheckDependencies(bpy.types.Operator):
    bl_idname = "blenderphotonics.check_deps"
    bl_label = "Check Dependencies"
    bl_description = "Check which dependencies are installed and available"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # Check current dependency status
        deps = check_dependencies()
        missing = [dep for dep, available in deps.items() if not available]
        available = [dep for dep, available in deps.items() if available]

        if missing:
            message = f"Missing: {', '.join(missing)}\nAvailable: {', '.join(available) if available else 'None'}"
            show_error_message(message, "Dependency Status")
        else:
            show_error_message("All dependencies are available!", "Dependency Status")

        return {"FINISHED"}
