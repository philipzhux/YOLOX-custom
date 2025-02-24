import json
import os
import subprocess
import signal
import time
from threading import Thread
from utils import load_json_atomic, file_lock, update_json_atomic

# Get paths from environment
CONFIG_FILE = os.getenv("CONFIG_FILE", "../config/config.json")
STATE_FILE = os.getenv("STATE_FILE", "../config/training_state.json")
EXP_FILE = os.getenv("EXP_FILE", "/home/chenz1/toorange/TBtest/YOLOX/exps/example/custom/my_yolox_gender.py")
TRAIN_TOOL_PATH = os.getenv("TRAIN_TOOL_PATH", "YOLOX-custom/tools/train.py")

# Global variables
training_process = None

def start_training():
    """Start the training process"""
    global training_process
    
    if training_process is not None and training_process.poll() is None:
        print("Training process already running")
        return
    
    config = load_json_atomic(CONFIG_FILE)
    # Update state with config values before starting
    # and unset the learning rate so that the trainer
    # can adapt it dynamically
    with file_lock(CONFIG_FILE) as f:
        config = json.load(f)
        if "learning_rate" in config:
            del config["learning_rate"]
        f.seek(0)
        f.truncate()
        json.dump(config, f, indent=2)
        
    update_json_atomic(STATE_FILE, {
        "stdout": "",
        "batch_size": config.get("batch_size", 64),
    })

    # Prepare training command
    cmd = [
        "python", TRAIN_TOOL_PATH,
        "-f", EXP_FILE,
        "-b", str(config.get("batch_size", 64)),
        "--fp16",
        "--logger", "tensorboard",
        "-d", "1",  # Use single GPU
    ]
    
    state = load_json_atomic(STATE_FILE)
    
    # Add resume flag and checkpoint if available
    if state.get("last_checkpoint") and os.path.exists(state["last_checkpoint"]):
        cmd.extend(["--resume", "-c", state["last_checkpoint"]])
        print(f"Resuming from checkpoint: {state['last_checkpoint']}")
    
    try:
        # Pass environment variables to subprocess
        env = os.environ.copy()
        env.update({
            "YOLOX_CONFIG_PATH": CONFIG_FILE,
            "YOLOX_STATE_PATH": STATE_FILE,
            "YOLOX_EXP_FILE": EXP_FILE,
            "YOLOX_ENABLE_DYNAMIC_CONFIG": "1"
        })
        
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=env,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Start stdout monitoring in a separate thread
        def monitor_output():
            while training_process and training_process.poll() is None:
                output = training_process.stdout.readline()
                if output:
                    current_state = load_json_atomic(STATE_FILE)
                    current_stdout = current_state.get("stdout", "")
                    update_json_atomic(STATE_FILE, {
                        "stdout": current_stdout + output
                    })
            
            # If process ended, ensure running state is False
            if training_process.poll() is not None:
                update_json_atomic(STATE_FILE, {
                    "running": False,
                    "stdout": load_json_atomic(STATE_FILE).get("stdout", "") + "\nTraining stopped.\n"
                })
                update_json_atomic(CONFIG_FILE, {
                    "running": False
                })
        
        Thread(target=monitor_output, daemon=True).start()
        
    except Exception as e:
        print(f"Error starting training: {e}")
        update_json_atomic(STATE_FILE, {
            "running": False,
            "error": str(e)
        })

def stop_training():
    """Stop the training process"""
    global training_process
    
    if training_process is None or training_process.poll() is not None:
        print("No training process to stop")
        return
    
    try:
        # Send SIGTERM to the process group
        os.killpg(os.getpgid(training_process.pid), signal.SIGTERM)
        
        # Wait for process to terminate
        training_process.wait(timeout=30)
        
        # Update state
        update_json_atomic(STATE_FILE, {
            "running": False,
            "stdout": load_json_atomic(STATE_FILE).get("stdout", "") + "\nTraining stopped.\n"
        })
        
        training_process = None
        
    except subprocess.TimeoutExpired:
        # If process doesn't terminate, force kill
        os.killpg(os.getpgid(training_process.pid), signal.SIGKILL)
        update_json_atomic(STATE_FILE, {
            "running": False,
            "stdout": load_json_atomic(STATE_FILE).get("stdout", "") + "\nTraining force stopped.\n"
        })
        training_process = None
    except Exception as e:
        print(f"Error stopping training: {e}")
        update_json_atomic(STATE_FILE, {
            "error": str(e)
        })

def get_state():
    """Get the current training state"""
    return load_json_atomic(STATE_FILE)

def update_state(state):
    """Update the training state"""
    update_json_atomic(STATE_FILE, state)

def consume_config():
    """Read and return the current config"""
    return load_json_atomic(CONFIG_FILE)

def monitor_config():
    """Monitor config file for running state changes"""
    while True:
        try:
            config = load_json_atomic(CONFIG_FILE)
            state = load_json_atomic(STATE_FILE)
            
            config_running = config.get("running", False)
            state_running = state.get("running", False)
            
            # Convert string to boolean if needed
            if isinstance(config_running, str):
                config_running = config_running.lower() == "true"
            
            # Start training if config says run but state says not running
            if config_running and not state_running:
                print("Config triggered training start")
                start_training()
            # Stop training if config says stop and state says running
            elif not config_running and state_running:
                print("Config triggered training stop")
                stop_training()
                
        except Exception as e:
            print(f"Error in config monitoring: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    # Start the config monitoring thread
    monitor_thread = Thread(target=monitor_config, daemon=True)
    monitor_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping train controller...")
        if training_process is not None:
            stop_training()
        update_json_atomic(STATE_FILE, {"running": False})
        update_json_atomic(CONFIG_FILE, {"running": False})

    
