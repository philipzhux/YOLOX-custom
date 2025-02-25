from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import time
from threading import Thread
from tensorflow.python.summary.summary_iterator import summary_iterator
from utils import load_json_atomic, save_json_atomic, update_json_atomic

app = Flask(__name__)
# Allow all origins for local dev
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    ping_timeout=60,  # Increase ping timeout
    ping_interval=25,  # Decrease ping interval
    async_mode='threading'  # Use threading mode
)

CONFIG_FILE = os.getenv("YOLOX_CONFIG_PATH", os.path.abspath("../config/config.json"))
STATE_FILE = os.getenv("YOLOX_STATE_PATH", os.path.abspath("../config/training_state.json"))
LOG_DIR = os.getenv("LOG_DIR", "/home/chenz1/toorange/TBtest/YOLOX/YOLOX_outputs/my_yolox_gender")

# Track processed event files
processed_mtimes = {}

def get_event_files():
    """Get all tensorboard event files sorted by modification time"""
    event_files = []
    for root, _, files in os.walk(LOG_DIR):
        for file in files:
            if "events.out.tfevents" in file:
                file_path = os.path.join(root, file)
                event_files.append((file_path, os.path.getmtime(file_path)))
    return sorted(event_files, key=lambda x: x[1])  # Sort by mtime

def parse_all_scalar_data():
    """Parse all tensorboard logs and return complete scalar data"""
    scalar_data = {}
    event_files = get_event_files()
    
    for file_path, _ in event_files:
        try:
            for event in summary_iterator(file_path):
                for value in event.summary.value:
                    if value.tag not in scalar_data:
                        scalar_data[value.tag] = []
                    scalar_data[value.tag].append({
                        "step": event.step,
                        "value": value.simple_value
                    })
        except Exception as e:
            print(f"Error reading event file {file_path}: {e}")
    
    # Sort each tag's data by step
    for tag in scalar_data:
        scalar_data[tag].sort(key=lambda x: x["step"])
    
    return scalar_data

def check_tensorboard_updates():
    """Monitor tensorboard logs for updates"""
    global processed_mtimes
    
    while True:
        try:
            # Get sorted list of event files
            event_files = get_event_files()
            
            # Check if any file is new or updated
            has_updates = False
            for file_path, mtime in event_files:
                if file_path not in processed_mtimes or processed_mtimes[file_path] < mtime:
                    has_updates = True
                    processed_mtimes[file_path] = mtime
            
            # If there are updates, emit complete scalar data
            if has_updates:
                scalar_data = parse_all_scalar_data()
                if scalar_data:  # Only emit if there's data
                    socketio.emit('scalar_update', scalar_data)
                    
        except Exception as e:
            print(f"Error in tensorboard monitoring: {e}")
            
        time.sleep(1)  # Check every second


def check_state_changes():
    last_mtime = 0
    while True:
        try:
            current_mtime = os.path.getmtime(STATE_FILE)
            if current_mtime != last_mtime:
                print("State change detected, emitting...")
                current_state = load_json_atomic(STATE_FILE)
                current_config = load_json_atomic(CONFIG_FILE)
                
                socketio.emit('state_update', {
                    'state': current_state,
                    'config': current_config
                })
                
                last_mtime = current_mtime
        except Exception as e:
            print(f"Error in state check: {e}")
        time.sleep(1)


# Ensure config.json exists
if not os.path.exists(CONFIG_FILE):
    default_config = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "running": False
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(default_config, f, indent=2)

# Ensure training_state.json exists
if not os.path.exists(STATE_FILE):
    default_state = {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epoch": 0,
        "stdout": "No logs yet.",
        "running": False
    }
    with open(STATE_FILE, "w") as f:
        json.dump(default_state, f, indent=2)

def parse_tensorboard_logs(log_dir):
    scalar_data = {}
    # Recursively walk logs dir to find any events.out.tfevents.* files
    for root, _, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                file_path = os.path.join(root, file)
                # read through all events, extracting scalars
                for event in summary_iterator(file_path):
                    for value in event.summary.value:
                        if value.tag not in scalar_data:
                            scalar_data[value.tag] = []
                        scalar_data[value.tag].append({
                            "step": event.step,
                            "value": value.simple_value
                        })
    return scalar_data

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

@socketio.on('update_config')
def handle_config_update(new_config):
    try:
        print("Received config update:", new_config)
        
        # Read current config first
        current_config = load_json_atomic(CONFIG_FILE)
        print("Current config:", current_config)
            
        # Update with new values
        current_config.update(new_config)
        print("Updated config:", current_config)
            
        # Save updated config with forced sync
        save_json_atomic(CONFIG_FILE, current_config)
            
        # Read current state to send back to client
        current_state = load_json_atomic(STATE_FILE)
            
        # Send both updated config and current state
        emit('state_update', {
            'state': current_state,
            'config': current_config
        })
            
        # Send success response to the client
        emit('config_updated', {'status': 'success'})
    except Exception as e:
        print("Error in config update:", str(e))
        emit('config_updated', {'status': 'error', 'message': str(e)})

@socketio.on('request_state')
def handle_state_request():
    try:
        state = load_json_atomic(STATE_FILE)
        config = load_json_atomic(CONFIG_FILE)
        emit('state_update', {'state': state, 'config': config})
    except Exception as e:
        emit('state_error', {'message': str(e)})

@app.route("/scalars", methods=["GET"])
def get_scalars():
    scalars = parse_tensorboard_logs(LOG_DIR)
    return jsonify(scalars)

if __name__ == "__main__":
    # Create the config and state files if they don't exist
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(STATE_FILE):
        with open(STATE_FILE, "w") as f:
            json.dump({}, f)
    # Start the monitoring thread
    monitor_thread = Thread(target=check_state_changes, daemon=True)
    monitor_thread.start()
    
    # Start Flask with SocketIO
    socketio.run(app, 
        debug=True, 
        port=5100,
        use_reloader=False  # Disable reloader in debug mode
    )