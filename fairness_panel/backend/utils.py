import json
import os
import fcntl
from contextlib import contextmanager

@contextmanager
def file_lock(path):
    """Context manager for file locking"""
    with open(path, 'r+') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield f
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

def load_json_atomic(path):
    """Load JSON file with file locking"""
    if not os.path.exists(path):
        return {}
    with file_lock(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {path}: {e}")
            return {}

def save_json_atomic(path, data):
    """Save JSON file with file locking"""
    with file_lock(path) as f:
        json.dump(data, f, indent=2)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())

def update_json_atomic(path, updates):
    """Update JSON file with file locking"""
    with file_lock(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
        data.update(updates)
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
        f.flush()
        os.fsync(f.fileno()) 