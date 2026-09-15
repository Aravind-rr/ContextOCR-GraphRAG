import os
import tempfile
from pathlib import Path

# Set before test modules import the application singleton.
test_database = Path(tempfile.gettempdir()) / "contextocr-test.db"
test_database.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{test_database.as_posix()}"
