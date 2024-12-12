
# TeleIoT-Fraud

## Overview
TeleIoT-Fraud process, analyse, and model telemetry data for fraud investigation from IoT devices.

The project contains various modules and scripts to process raw data, perform feature engineering, and create predictive models. It also includes notebooks for exploratory data analysis (EDA) and temporal sampling, as well as utility functions to handle the data efficiently.

## Project Structure

```
├── LICENSE                        # License information for the project
├── Logs                           # Log files for tracking the project's progress
├── Models                         # Directory to store machine learning models
├── Notebooks                      # Jupyter Notebooks for experimentation and analysis
│   └── dev.ipynb                  # Development and testing notebook
├── RScripts                       # R scripts for complex data processing and analysis
│   ├── aggregate_json.R           # R script for aggregating JSON data
│   └── temporal_sampling.R        # R script for creating temporal samples
├── config.yml                     # Configuration file for paths and parameters
├── create_params_file.py          # Python script to generate parameter files
├── readme.md                      # Project documentation file (this file)
├── setup_env.sh                   # Automates setting up the project dependencies
├── src                            # Python source code for various modules
│   ├── eda.py                     # A class with over 20 functions for Exploratory Data Analysis on sensory data
│   └── enrich.py                  # Data enrichment module
└── unittest                       # Unit tests for verifying project components
    └── addGforce.py               # Unit test for adding G-force related calculations
```

## Features
- Data Preprocessing: Includes R and Python scripts for cleaning and transforming raw data into structured formats.
- Temporal Sampling: Implements sliding window and subsampling techniques for time-series data analysis.
- Model Training: Provides scripts for training machine learning models using the processed data.
- **Anonymised Data**: Due to current privacy concerns, the data is not available. However, anonymised data will be made available soon for research and analysis.

## Installation

### Prerequisites
Before using the TeleIoT-Fraud project, ensure you have the following installed:
- Python 3.x
- R (for R scripts)
- Jupyter Notebook (for interactive development)
- Required libraries (listed in requirements)

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/TeleIoT-Fraud.git
    ```

2. Install dependencies for Python and Creating TFR Environment:
    ```bash
    ./setup_env.sh
    ```

3. For R dependencies, run:
    ```R
    install.packages(c("data.table", "dplyr", "lubridate", "rio", "rmarkdown", "yaml"))
    ```

4. Configure the project by modifying the `config.yml` file to include your data paths and parameters.

## Data Privacy Notice
Currently, the raw data used for the project is unavailable due to privacy and confidentiality concerns. However, we plan to release anonymised datasets shortly. The anonymised data will be suitable for research and analysis purposes, and we will ensure that sensitive information is removed or masked to protect privacy.

## Usage
- Refer to the `Notebooks/dev.ipynb` for an example of how to perform exploratory data analysis and model training.
- The `RScripts` directory contains R scripts for data aggregation and temporal sampling tasks.
- The Python modules in the `src` directory are designed for enriching and processing the data.

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing
We welcome contributions to improve this project. Please feel free to open issues and submit pull requests.
