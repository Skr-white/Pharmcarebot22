# shared_state.py
import threading
from typing import Any, Dict, Optional

# Thread-safe shared data dictionary
shared_data: Dict[str, Any] = {}
lock = threading.Lock()

def update_state(key: str, value: Any) -> None:
    """Update a value in the shared state in a thread-safe way."""
    with lock:
        shared_data[key] = value

def get_state(key: str, default: Optional[Any] = None) -> Any:
    """Retrieve a value from the shared state in a thread-safe way."""
    with lock:
        return shared_data.get(key, default)