# shared_state.py
import threading

# Shared dictionary to store state between brains
shared_data = {}
lock = threading.Lock()

# Helper functions to get/update state safely
def update_state(key: str, value):
    with lock:
        shared_data[key] = value

def get_state(key: str, default=None):
    with lock:
        return shared_data.get(key, default)