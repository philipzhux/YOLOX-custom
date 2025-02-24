#!/bin/bash

# Change directory to backend
cd /home/chenz1/toorange/TBtest/YOLOX/fairness_panel/backend

# Run set_env_vars.sh
source /home/chenz1/toorange/TBtest/YOLOX/fairness_panel/set_env_vars.sh

# Run the application
python train_controller.py