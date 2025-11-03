# shared_state.py
import threading

# A thread-safe dictionary for shared state
state_lock = threading.Lock()
shared_state = {
    "last_user_message": None,
    "last_bot_response": None,
    "context_data": {}
}

def update_state(key, value):
    with state_lock:
        shared_state[key] = value

def get_state(key):
    with state_lock:
        return shared_state.get(key)