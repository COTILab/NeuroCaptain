"""Entry point for the NeuroCaptain test suite.

Run inside Blender's bundled Python:

    blender --background --factory-startup --python-exit-code 1 --python tests/run_tests.py

Discovers every tests/test_*.py module and runs it with stdlib unittest.
"""

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)


def _report_github_annotations(result):
    """Emit GitHub Actions ::error:: workflow commands for each failure/error.

    These show up directly on the run summary and PR checks tab, so a
    failure is visible without opening the step log. No-op outside GH Actions.
    """
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for test, trace in result.failures + result.errors:
        message = trace.strip().splitlines()[-1]
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0D%0A")
        print(f"::error title=NeuroCaptain test failed::{test}: {message}")


if __name__ == "__main__":
    suite = unittest.TestLoader().discover(TESTS_DIR, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    _report_github_annotations(result)
    if not result.wasSuccessful():
        sys.exit(1)
