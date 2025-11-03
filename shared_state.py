# shared_state.py
import threading

lock = threading.Lock()
shared_data = {
    "last_user_message": "",
    "last_bot_response": ""
}