"""Test configuration.

Ensures ``src/`` is importable even when pytest is invoked in a way that does
not pick up ``pytest.ini`` (e.g. running a single file by path).
"""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
