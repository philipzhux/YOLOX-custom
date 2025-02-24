from utils import load_json_atomic, save_json_atomic, update_json_atomic
import time
import random
import tensorflow as tf
import os

CONFIG_FILE = "./config.json"
STATE_FILE = "./training_state.json"
LOG_DIR = "./logs"

def simulate_training_loop():
    writer = tf.summary.create_file_writer(LOG_DIR)
    
    while True:
        # 1) Load config + state
        config = load_json_atomic(CONFIG_FILE)
        state = load_json_atomic(STATE_FILE)

        # 2) Merge config -> state
        updates = {
            "batch_size": config.get("batch_size", state.get("batch_size", 32)),
            "learning_rate": config.get("learning_rate", state.get("learning_rate", 0.001))
        }
        
        # Convert running to boolean if it's a string
        config_running = config.get("running", False)
        if isinstance(config_running, str):
            config_running = config_running.lower() == "true"
        updates["running"] = config_running

        # Only simulate training if running is True
        if updates["running"] is True:
            # 3) Simulate training: increment epoch, set random loss
            updates["epoch"] = state.get("epoch", 0) + 1
            loss = round(random.uniform(0, 1), 4)
            updates["loss_rate"] = loss

            # 4) Append log line
            old_logs = state.get("stdout", "")
            new_line = f"Epoch {updates['epoch']} => loss {loss}\n"
            # Keep logs from growing forever
            updates["stdout"] = (old_logs + new_line)[-1500:]

            # 5) Write to TensorBoard logs
            with writer.as_default():
                tf.summary.scalar("training_loss", loss, step=updates["epoch"])
                writer.flush()

        # Save updated state with forced sync
        update_json_atomic(STATE_FILE, updates)
        time.sleep(5)

if __name__ == "__main__":
    simulate_training_loop()