#!/bin/bash

################################################################################
# Script Name: setup_env.sh
# Description: Automates the creation and setup of the project environment and 
#              creates a python environment named TFR with all non-conflicting python libraries
#              specified in a python-lib text file. This script also logs the results of the 
#              installation process showing successful and faild installations. 
#
# Usage:
#   1. Place the script in the project directory containing "python-libs.txt".
#   2. Make the script executable: chmod +x setup_env.sh
#   3. Run the script: ./setup_env.sh
#
# Requirements:
#   - Conda must be installed and accessible in the system PATH.
#   - "python-libs.txt" file should list the required dependencies, one per line.
#
# Features:
#   - Automatically removes any existing environment with the same name.
#   - Creates a new Conda environment with the specified Python version.
#   - Installs packages listed in "python-libs.txt".
#   - Verifies Python version compatibility.
#   - Logs successes and failures for easy troubleshooting.
#
# Logs:
#   - Successful installations are logged in "installation_success.log".
#   - Failed installations are logged in "failed_installations.log".
#
# Author: Ashkan-Dashtban@2024 github.com/AshkanDn
# Date: 12/12/2024
# Version: 1.6
################################################################################
clear 
#source ~/.zshrc
conda deactivate

# Exit on error
set -euo pipefail

# Constants for better maintainability
readonly ENV_NAME="TFR"
readonly PYTHON_VERSION="3.10.4"
readonly IPYKERNEL_VERSION="6.29.0"
readonly REQUIREMENTS_FILE="python-libs.txt"
readonly FAILED_LOG="failed_installations.log"
readonly SUCCESS_LOG="installation_success.log"
readonly DATE_FORMAT="+%Y-%m-%d %H:%M:%S"
readonly START_TIME=$(date "$DATE_FORMAT")

# Function to log messages
log_message() {
    local message="$1"
    local log_type="$2" # INFO, ERROR, SUCCESS
    local timestamp=$(date "$DATE_FORMAT")

    echo "[$timestamp] $log_type: $message"

    case "$log_type" in
        ERROR)
            echo "[$timestamp] $log_type: $message" >> "$FAILED_LOG" ;;
        SUCCESS)
            echo "[$timestamp] $log_type: $message" >> "$SUCCESS_LOG" ;;
    esac
}

# Clear logs to start fresh
: > "$FAILED_LOG"
: > "$SUCCESS_LOG"
log_message "Installation script started." "INFO"

# Deactivate and remove existing Conda environment (if any)
source ~/.zshrc
conda deactivate

# Create a new Conda environment
log_message "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION." "INFO"
conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y
conda activate "$ENV_NAME" || { log_message "Failed to activate Conda environment: $ENV_NAME." "ERROR"; exit 1; }

# Install the IPython kernel
log_message "Installing IPython kernel for the environment." "INFO"
pip install ipykernel=="$IPYKERNEL_VERSION"
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"

# Set environment variable to disable file validation
export PYDEVD_DISABLE_FILE_VALIDATION=1

# Check if requirements.txt exists
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    log_message "Error: File '$REQUIREMENTS_FILE' not found. Exiting." "ERROR"
    exit 1
fi

# Loop through requirements and install packages
while IFS= read -r package || [[ -n "$package" ]]; do
    # Skip empty or commented lines
    [[ -z "$package" || "$package" =~ ^# ]] && continue

    log_message "Installing package: $package." "INFO"

    # Install the package
    if pip install "$package" &> /dev/null; then
        log_message "Successfully installed: $package." "SUCCESS"
    else
        log_message "Failed to install: $package." "ERROR"
    fi

done < "$REQUIREMENTS_FILE"

# Ensure Python version compatibility
log_message "Verifying Python version compatibility." "INFO"
conda install -y python="$PYTHON_VERSION" || {
    log_message "Failed to ensure Python version $PYTHON_VERSION." "ERROR"
    exit 1
}

# Summary and completion
END_TIME=$(date "$DATE_FORMAT")
log_message "Installation script completed. Started at $START_TIME and ended at $END_TIME." "INFO"
log_message "Failed installations logged in '$FAILED_LOG'." "INFO"
log_message "Successful installations logged in '$SUCCESS_LOG'." "INFO"

# Exit successfully
exit 0



