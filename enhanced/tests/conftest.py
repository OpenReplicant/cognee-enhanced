import os
import tempfile

# Importing cognee creates its default stores; keep them out of the repo root.
_root = tempfile.mkdtemp(prefix="enhanced-tests-")
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", os.path.join(_root, "system"))
os.environ.setdefault("DATA_ROOT_DIRECTORY", os.path.join(_root, "data"))
