"""
Project Configuration Setup Script:

This script initialises and organizes the project's directory structure,
creates configuration paths, and saves them in a YAML file in the Project Directory root.

@Ashkan Dashtban
"""

import os
import yaml
from datetime import datetime
from collections import OrderedDict

def create_config_yaml(project_dir, config_file):
    """
    Create a YAML configuration file for the project.

    Parameters:
        project_dir (str): Root directory of the project.
        config_file (str): Path to save the generated YAML configuration.

    Returns:
        None
    """
    try:
        # Ensure the working directory is set to the project root
        os.chdir(project_dir)
        print(f"Working directory set to: {os.getcwd()}")

        # Get current date and time
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Define directory paths
        path_params = OrderedDict({
            'main': project_dir,
            'log': os.path.join(project_dir, "Logs/"),
            'report': os.path.join(project_dir, "Reports/"),
            'data': os.path.join(project_dir, "Data/"),
            'sdata': os.path.join(project_dir, "SourceData/"),
            'modules': os.path.join(project_dir, "src/"),
            'R-scripts': os.path.join(project_dir, "RScripts/"),
            'notebooks': os.path.join(project_dir, "Notebooks/"),
            'maps': os.path.join(project_dir, "Data/Maps/"),
            'rdata': os.path.join(project_dir, "Data/Rdata/"),
            'current_datetime': current_time,
            'author': "Ashkan Dashtban",
            'updater': "Ashkan Dashtban",
        })

        # Define data file paths
        data = {
            'refined': os.path.join(project_dir, "Data/df_refined.pkl"),
            'refined_enriched': os.path.join(project_dir, "Data/df_refined_enriched.pkl"),
            'rdata_scanned': os.path.join(project_dir, "Data/Rdata/processed_data_scanned.csv"),
            'rdata_events': os.path.join(project_dir, "Data/Rdata/processed_data_with_events.csv"),
            'rdata_noevent': os.path.join(project_dir, "Data/Rdata/processed_data_no_events.csv"),
            'rdata_complete': os.path.join(project_dir, "Data/Rdata/processed_data_complete.csv"),
        }

        # Define report paths
        reports = {
            'maps': os.path.join(project_dir, "Reports/maps/"),
        }

        # Compile all configuration into a dictionary
        yml_contents = {
            'path': path_params,
            'reports': reports,
            'data': data,
        }

        # Save the configuration as a YAML file
        with open(config_file, 'w') as yaml_file:
            yaml.dump(yml_contents, yaml_file, default_flow_style=False)

        print(f"Configuration file saved to: {config_file}")

    except Exception as e:
        print(f"An error occurred while creating the configuration file: {str(e)}")
        raise

# Define project directory and configuration file path
PROJECT_DIR = './dashtban/TeleIoT-Fraud/'
CONFIG_FILE = os.path.join(PROJECT_DIR, "config.yml")

# Generate the configuration YAML
create_config_yaml(PROJECT_DIR, CONFIG_FILE)
