"""
Module: dataloader
Written by: @Ashkan Dashtban (Dashtban.edu@gmail.com) "2022-2024"


This module provides flexible and user-friendly data frame -based input/output (I/O) operations designed for the development phase. 
It offers two intelligent functions, `load` and `save`, which handle various data formats with customizable input parameters.

The `load` function automatically detects file types or searches for multiple file extensions given a single file name. 
It can search across specified directories or default paths and offers options to limit the number of rows loaded or load 
a fraction of the data. The function also allows for displaying a subset of rows, and can handle arguments in any order, 
making it highly flexible and intuitive to use.The `save` function automatically detects the appropriate file format and supports overwriting existing files, providing 
seamless saving capabilities for a variety of data structures.

Both functions incorporate logging for status updates and raise custom exceptions in case of errors, ensuring robust error handling.


Functions:
    - load: Loads data from a file (CSV, TSV, Excel, JSON, or Pickle) into a pandas DataFrame or Pickle object. 
            Supports loading a fraction of the file or displaying a subset of rows.
    - save: Saves data (pandas DataFrame, list, dict, or other serializable objects) to disk in various formats 
            (CSV, JSON, Excel, or Pickle).

Constants:
    - DEFAULT_DIRECTORY: The default directory (`data`) where files are searched for or saved.
    - DEFAULT_FILE_TYPE: The default file type (`csv`) used when no extension is specified.
"""



import os
import pandas as pd
import logging
import pickle
from typing import Any, Optional, Union
import random  


# Constants
DEFAULT_DIRECTORY = "data"
DEFAULT_FILE_TYPE = "csv"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Custom Exception Class -- to be completed
class DataIOException(Exception):
    """Custom exception for errors related to file I/O operations."""
    pass

def get_last_dir(full_path: str) -> str:
	# Extract the last directory and file name
	return(
		os.path.join(
    		".",  # Represent the current directory
   	 		os.path.basename(os.path.dirname(full_path)),  # Get the last directory name
    		os.path.basename(full_path)
        )
    )

def get_file_type(file_name: str) -> str:
    """Extract the file type based on the extension."""
    _, ext = os.path.splitext(file_name)
    return ext.lower().lstrip('.') if ext else None


def ensure_directory_exists(directory: str) -> None:
    """Ensure the specified directory exists."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
        
def load(
    file_name: str, 
    data_directory: Optional[Union[str, int, float]] = None, 
    show: Optional[Union[int, float]] = None, 
    prop: Optional[Union[int, float]] = None
) -> Optional[pd.DataFrame]:    
    """
    Loads a pandas DataFrame or a Pickle object from a file, supporting various formats such as CSV, TSV, Excel, JSON, and Pickle. 
    Optionally limits the number of rows displayed based on a specified row count (`show`) and optionally loads a fraction of the file based on a proportion (`prop`).

    Supported Formats:
        - CSV, TSV, Excel, JSON, Pickle, PKL (case-insensitive).
        - If no file extension is provided, the function will search for the file in the specified directory (if provided), 
          or in the `./data` and `./results` directories. It will attempt to load the file with any supported extension and inform the user about which file was loaded.

    Use Cases:
        - Load the entire file:
          `load(df_2_days_ago)`
        
        - Load the entire data and shows specified number of rows:
          `load(df, 5)`  # Loads the entire file and displays the top 5 rows.
        
        - Load a fraction of the file:
          `load(my_huge_file, 0.05)`  # Loads 5% of a large file randomly.

        - Load a fraction of a file and show top rows:
          `load(my_huge_file, 10, 0.05)`  # Loads 5% randomly and displays the top 10 rows.
          `load(my_huge_file, 0.05, 10)`  # The same result as above.

        - Load from a specific directory and show top rows:
          `load(my_df, my_directory, 10)`  # Loads the entire file and displays the top 10 rows from a file in `my_directory`.
        
        - Load a fraction from a specific directory and show top rows:
          `load(my_df, my_directory, 10, 0.05)`  # Loads 5% of the file and shows the top 10 rows.

    Parameters:
        file_name (str): The name of the file to load.
        data_directory (str, optional): The directory where the file is located. If None, the function defaults to the directory specified by 
                                        `os.path.join(os.getcwd(), DEFAULT_DIRECTORY)`.
        show (int, optional): The number of rows to display from the file. If provided, it will limit the output to the specified number of rows.
        prop (float, optional): A fraction (0 < prop < 1) of the file to load. If provided, it will limit the amount of the file to load based on this fraction.

    Returns:
        pd.DataFrame: The loaded DataFrame, optionally displaying the top rows based on the `show` parameter.

    Logic:
        - If the second argument (`data_directory`) is a float or integer and `show` is not provided:
            - If it’s a float between 0 and 1, it will be treated as a fraction (`prop`) of the file to load.
            - If it’s an integer greater than 1, it will be treated as the number of rows to load (`show`).
        - If `data_directory` is None, the function will automatically use the default directory (`os.path.join(os.getcwd(), DEFAULT_DIRECTORY)`).
        - If `show` is a float between 0 and 1, it will be treated as the proportion of the file to load (via `prop`), and `show` will be ignored.
        - If `show` is an integer greater than 1, it will be treated as the number of rows to load from the file.

    Example:
        load("data_file", show=10)  # Loads the entire file and shows the first 10 rows.
        load("large_file", prop=0.10)  # Loads 10% of the file randomly.
        load("file_in_directory", data_directory="my_dir", show=20)  # Loads the entire file and shows the top 20 rows from a specific directory.
    """
    # Handle Flexcible orders
    if isinstance(data_directory, float):
        if isinstance(prop, int):
            show = prop
            prop = data_directory
            data_directory = None
        else:
            prop = data_directory
            data_directory = None
    
    if isinstance(data_directory, int):
        if isinstance(show, float):
            prop = show
            show = data_directory
            data_directory = None
        else:
            show = data_directory
            data_directory = None  
    
    if isinstance(show, float):
        if isinstance(prop, int):
            temp = show
            show = prop
            prop =  temp
        else:
            prop = show
            show = None

    data_directory = data_directory or os.path.join(os.getcwd(), DEFAULT_DIRECTORY)
    secondary_directory = os.path.join(os.getcwd(), "results")
    
    
    try:
        # Ensure the primary directory exists
        ensure_directory_exists(data_directory)

        # Construct potential file paths to try in both directories
        file_paths_to_try = [os.path.join(data_directory, file_name)]
        if not get_file_type(file_name):  # If no extension is provided
            file_paths_to_try.extend([
                os.path.join(data_directory, f"{file_name}.csv"),
                os.path.join(data_directory, f"{file_name}.tsv"),
                os.path.join(data_directory, f"{file_name}.pkl"),
                os.path.join(data_directory, f"{file_name}.pickle"),
                os.path.join(data_directory, f"{file_name}.xls"),
                os.path.join(data_directory, f"{file_name}.xlsx"),
                os.path.join(data_directory, f"{file_name}.json"),
            ])
        
        # Add paths from secondary directory
        file_paths_to_try.extend(
            [path.replace(data_directory, secondary_directory) for path in file_paths_to_try]
        )

        # Set prop and show values based on inputs
        if isinstance(show, float) and 0 < show < 1:
            prop, show = show, None
        elif isinstance(show, int) and show >= 1:
            prop = None
        
        # Attempt to load the file from constructed paths
        df, file_type = None, None
        for file_path in file_paths_to_try:
            if os.path.exists(file_path):
                file_type = get_file_type(file_path)

                # Load file based on type
                if file_type == "csv":
                    if prop:
                        n = sum(1 for _ in open(file_path)) - 1  # Total rows excluding header
                        skip_indices = set(random.sample(range(1, n + 1), int(n * (1 - prop))))
                        df = pd.read_csv(file_path, low_memory=False, skiprows=lambda x: x in skip_indices)
                    else:
                        df = pd.read_csv(file_path, low_memory=False)
                elif file_type == "tsv":
                    if prop:
                        n = sum(1 for _ in open(file_path)) - 1
                        skip_indices = set(random.sample(range(1, n + 1), int(n * (1 - prop))))
                        df = pd.read_csv(file_path, sep="\t", low_memory=False, skiprows=lambda x: x in skip_indices)
                    else:
                        df = pd.read_csv(file_path, sep="\t", low_memory=False)
                elif file_type in ["pickle", "pkl"]:
                    with open(file_path, "rb") as f:
                        df = pickle.load(f)
                elif file_type in ["xls", "xlsx"]:
                    df = pd.read_excel(file_path, nrows=int(prop * 1000) if prop else None)
                elif file_type == "json":
                    df = pd.read_json(file_path, orient="records", lines=True)
                else:
                    raise DataIOException(
                        f"Unsupported file type: '{file_type}'. Supported types are 'csv', 'tsv', 'excel', 'json', 'pickle'."
                    )
                break

        if df is not None:
            logging.info(f"'{file_name}.{file_type}' loaded with shape:{df.shape}")
            if show:
                display(df.head(show))
            return df
        else:
            raise FileNotFoundError(f"File '{file_name}' not found in specified directories or with supported extensions.")

    except Exception as e:
        logging.error(f"Error loading file '{file_name}': {e}")
   

def save(
    data: Union[pd.DataFrame, list, dict, Any],
    file_name: str,
    data_directory: Optional[str] = None
) -> None:
    """
    Save data to disk in a format determined by its structure or specified by the user.

    Parameters
    ----------
    data : pd.DataFrame, list, dict, or other serializable objects
        The data to save.
    file_name : str
        The file name or path where the file should be saved.
    data_directory : str, optional
        The directory where the file should be saved. Defaults to 'data' folder in the current working directory.
    """
    data_directory = data_directory or os.path.join(os.getcwd(), DEFAULT_DIRECTORY)

    try:
        # Ensure the directory exists
        ensure_directory_exists(data_directory)

        # Determine the full file path and file extension
        file_path = os.path.join(data_directory, file_name)
        file_type = get_file_type(file_name)

        # Automatically assign file type based on data type if no file type is specified
        if not file_type:
            if isinstance(data, pd.DataFrame):
                file_type = 'csv'
                file_path += '.csv'
            elif isinstance(data, (list, dict)):
                file_type = 'json'
                file_path += '.json'
            else:
                file_type = 'pkl'
                file_path += '.pkl'

        # Check if the file already exists and prompt before overwriting
        if os.path.exists(file_path):
            user_input = input(f"'{get_last_dir(file_path)}' already exists. Overwrite? (y/n): ").strip().lower()
            if user_input != 'y':
                logging.info("Save operation cancelled.")
                return

        # Save based on data structure and type
        if isinstance(data, pd.DataFrame):
            if file_type == 'csv':
                data.to_csv(file_path, index=False)
                logging.info(f"DataFrame successfully saved to '{get_last_dir(file_path)}' as CSV.")
            elif file_type in ['xls', 'xlsx', 'excel']:
                data.to_excel(file_path, index=False)
                logging.info(f"DataFrame successfully saved to '{get_last_dir(file_path)}' as Excel.")
            elif file_type == 'json':
                data.to_json(file_path, orient='records', lines=True)
                logging.info(f"DataFrame successfully saved to '{get_last_dir(file_path)}' as JSON.")
            elif file_type in ['pickle', 'pkl']:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                logging.info(f"DataFrame successfully saved to '{get_last_dir(file_path)}' as Pickle.")
            else:
                raise DataIOException(f"Unsupported file type for DataFrame: '{file_type}'.")
        elif isinstance(data, (list, dict)):
            if file_type == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                logging.info(f"Data successfully saved to '{get_last_dir(file_path)}' as JSON.")
            elif file_type in ['pickle', 'pkl']:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                logging.info(f"Data successfully saved to '{get_last_dir(file_path)}' as Pickle.")
            else:
                raise DataIOException(f"Unsupported file type for list/dict: '{get_last_dir(file_path)}'. Use JSON or Pickle.")
        else:
            # Fallback: Save unsupported types as Pickle
            if file_type not in ['pickle', 'pkl']:
                file_path = os.path.splitext(file_path)[0] + '.pkl'
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
            logging.info(f"Data of type '{type(data).__name__}' successfully saved to '{get_last_dir(file_path)}' as Pickle.")

    except Exception as e:
        raise DataIOException(f"Error saving data: {e}") from e

