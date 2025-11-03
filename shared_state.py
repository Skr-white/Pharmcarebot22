import threading

lock = threading.Lock()
shared_data = {
    "last_user_message": "",
    "last_bot_response": ""
}

def update_state(key: str, value: str):
    """Update a value in the shared state safely."""
    with lock:
        shared_data[key] = value

def get_state(key: str):
    """Get a value from the shared state safely."""
    with lock:
        return shared_data.get(key, "")