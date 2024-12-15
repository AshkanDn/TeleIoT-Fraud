
"""
EDA Class - Exploratory Data Analysis for Geospatial and Sensor Data
====================================
This Class includes a robust set of functions designed to facilitate exploratory data analysis (EDA) for telemetry and geospatial datasets. It assists with data cleaning, visualization, statistical analysis, and the identification of patterns, outliers, and correlations that can drive deeper insights for further analysis or model development. 

@Lead Analyst: Ashkan Dashtban

Key Key Functionalities:
- **Data Summary and Visualization**: Generate summaries, statistical reports, and interactive maps for journey and sensor data.
- **Handling Missing Data**: Identify, select, and manage missing data, including generating missing value indicators and reporting.
- **Correlation Analysis**: Calculate and visualize correlations across multiple features.
- **Geospatial Map Generation**: Create interactive maps for journeys, displaying critical telemetry data such as speed and HDOP.
- **Statistical Reporting**: Generate detailed HTML reports with summary statistics and insights.

Functions:

1. **tab(df: pd.DataFrame, *columns: str) -> Union[dict, None]**:
   - Displays a summary of specified columns or all columns in the DataFrame, aiding in initial data inspection.

2. **generate_maps(df: pd.DataFrame, output_dir: str = "maps/", journey_id_column: str = "journeyId", 
                 lat_column: str = "coordinates.lat", lon_column: str = "coordinates.lon", 
                 hdop_column: str = "hdop", speed_column: str = "speed", 
                 speed_mph_column: str = "speedMph")**:
   - Generates interactive maps for each journey ID in the DataFrame, overlaying telemetry data such as HDOP and speed.

3. **relative_ranges(df: pd.DataFrame, base_column: str, output_file: Union[str, None] = None) -> pd.DataFrame**:
   - Computes and optionally saves the relative ranges of values in a specified base column, providing insights into variability.

4. **generate_correlation_heatmap(dataframe: pd.DataFrame, output_file: Union[str, None] = None, 
                                x_rotation: int = 90, y_rotation: int = 0) -> Union[str, None]**:
   - Generates a heatmap to visualize correlations between features in the DataFrame, with an option to save to a file.

5. **generate_correlation_table(dataframe: pd.DataFrame, output_file: Union[str, None] = None) -> Union[pd.DataFrame, None]**:
   - Produces a table displaying correlations between columns in the DataFrame, useful for understanding relationships within the data.

6. **select_random_rows_without_missing_values(df: pd.DataFrame) -> pd.DataFrame**:
   - Selects a random sample of rows from the DataFrame, excluding those with missing values, for unbiased analysis.

7. **generate_html_stats(dataframe: pd.DataFrame, output_file: str) -> None**:
   - Generates an HTML report containing summary statistics for the DataFrame, facilitating easy inspection of key metrics.

8. **summary_html(dataframe: pd.DataFrame, output_file: Union[str, None] = None) -> Union[pd.DataFrame, None]**:
   - Summarizes the DataFrame and optionally saves the summary as an HTML file, providing a quick overview of the data.

9. **select_rows_with_missing_values(df: pd.DataFrame, output_file: Optional[str] = None) -> Union[pd.DataFrame, None]**:
   - Identifies and optionally saves rows containing missing values, allowing for focused cleaning efforts.

10. **convert_to_nearest_datatype(df: pd.DataFrame) -> pd.DataFrame**:
   - Converts columns to their most appropriate data types, optimizing the DataFrame for analysis.

11. **show_column_types(df: pd.DataFrame) -> None**:
   - Displays the data types of all columns in the DataFrame, helping with type validation.

12. **add_missing_indicator_column(df: pd.DataFrame, threshold: float = 0.8, colname: str = 'missing') -> pd.DataFrame**:
   - Adds an indicator column to the DataFrame, flagging rows with missing values above a specified threshold.

13. **null_rows_with_non_null_column(null_column: str, non_null_column: str, df: pd.DataFrame) -> Optional[pd.DataFrame]**:
   - Selects rows where a given column has null values while another column is non-null, helping identify inconsistent data.

14. **relative_non_null(df: pd.DataFrame, base_column: str, output_file: Union[str, None] = None) -> pd.DataFrame**:
   - Calculates the relative count of non-null values for a specified base column, optionally saving the results to a file.

15. **column_statistics(df: pd.DataFrame, output_html: Union[str, None] = None) -> pd.DataFrame**:
   - Computes basic statistics (e.g., mean, median, standard deviation) for each column in the DataFrame, with optional HTML output.

16. **read_files(data_directory: str) -> tuple**:
   - Reads JSON files from a specified directory into a DataFrame, returning the DataFrame, any errors encountered, and the list of loaded files.

17. **remove_columns_rows_with_single_value_or_null(df: pd.DataFrame, output_file: Optional[str] = None, 
                                                 threshold_col: float = 0.5, threshold_row: float = 0.75)**:
   - Removes columns with a single value or excessive null values, and rows with excessive nulls. Optionally generates a summary report.


Important Notes:
- Ensure that the input DataFrame contains the necessary columns for each function to operate correctly.
- Functions are designed to be flexible and can handle various data formats; however, pre-processing for missing or inconsistent data will improve performance.

"""


import pandas as pd
import os
import json
from collections import defaultdict
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
import matplotlib.pyplot as plt
import folium

# Initialise logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EDA:
    """
    EDA Class - Exploratory Data Analysis for Geospatial and Sensor Data
    ================================================================
    This class provides a robust set of tools designed to facilitate exploratory data analysis (EDA) 
    for telemetry and geospatial datasets. It assists with data cleaning, visualization, statistical analysis,
    and identification of patterns, outliers, and correlations.

    Functions:
    - Data Summary and Visualization
    - Handling Missing Data
    - Correlation Analysis
    - Geospatial Map Generation
    - Statistical Reporting
    """

    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize the EDA class.

        Parameters:
        df (pandas.DataFrame, optional): The DataFrame to be analyzed. Defaults to None.
        """
        self.df = df  # DataFrame is optional at initialization
        if df is not None:
            self._check_dataframe(df)
        
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        Set or update the DataFrame in the EDA class.

        Parameters:
        df (pandas.DataFrame): The new DataFrame to be set.
        """
        self.df = df
        self._check_dataframe(df)
    
    def _check_dataframe(self, df: pd.DataFrame) -> None:
        """
        Checks if the provided DataFrame is valid.

        Parameters:
        df (pandas.DataFrame): The DataFrame to be checked.
        
        Raises:
        ValueError: If the DataFrame is empty or invalid.
        """
        if df.empty:
            raise ValueError("The provided DataFrame is empty.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The provided input is not a pandas DataFrame.")

    def tab(self, *columns: str) -> Union[dict, None]:
        """
        Tabulate the values of specified columns in the DataFrame.

        - If one column is specified: Returns a DataFrame with value counts and proportions for that column.
        - If two columns are specified: Returns a cross-tabulation DataFrame showing counts for each combination of values in the two columns.

        Parameters:
            *columns (str): Column names to be tabulated.

        Returns:
            dict: A dictionary containing the tabulated results for each column.
        """
        tabulation = {}
        columns_not_found = []
        
        # Filter out columns not present in DataFrame
        valid_columns = [col for col in columns if col in self.df.columns]
        columns_not_found = [col for col in columns if col not in self.df.columns]

        if valid_columns:
            for col in valid_columns:
                if len(valid_columns) == 1:  # Single column: Value counts and proportions
                    counts = self.df[col].value_counts()
                    proportion = (counts / len(self.df) * 100).round(1)
                    tabulation[col] = pd.DataFrame({'Count': counts, 'Proportion (%)': proportion})
                elif len(valid_columns) > 1:  # Multiple columns: Cross-tabulation
                    tabulation[col] = self.df.groupby(valid_columns[0])[valid_columns[1]].value_counts().unstack(fill_value=0)

        if columns_not_found:
            print(f"The following columns were not found in the DataFrame: {', '.join(columns_not_found)}")

        return tabulation if valid_columns else None

    def generate_maps(self, output_dir: str = "maps/", journey_id_column: str = "journeyId", 
                      lat_column: str = "coordinates.lat", lon_column: str = "coordinates.lon", 
                      hdop_column: str = "hdop", speed_column: str = "speed", 
                      speed_mph_column: str = "speedMph") -> None:
        """
        Generates a map for each journey in the DataFrame and saves it as an HTML file.
        """
        if journey_id_column not in self.df.columns:
            raise ValueError(f"Column '{journey_id_column}' not found in the DataFrame.")
        
        os.makedirs(output_dir, exist_ok=True)

        distinct_journey_ids = self.df[journey_id_column].unique()

        for journey_id in distinct_journey_ids:
            try:
                journey_df = self.df[self.df[journey_id_column] == journey_id]
                map_ = folium.Map(location=[journey_df[lat_column].iloc[0], journey_df[lon_column].iloc[0]], zoom_start=15)

                # Add markers with telemetry data
                for lat, lon, hdop, speed, speed_mph in zip(journey_df[lat_column], journey_df[lon_column], 
                                                             journey_df[hdop_column], journey_df[speed_column], 
                                                             journey_df[speed_mph_column]):
                    popup = f"HDOP: {hdop}<br>Speed: {speed} km/h<br>Speed (mph): {speed_mph} mph"
                    folium.Marker(location=[lat, lon], popup=popup).add_to(map_)

                # Add a polyline for the journey path
                path_points = list(zip(journey_df[lat_column], journey_df[lon_column]))
                folium.PolyLine(path_points, color="blue", weight=2.5, opacity=1).add_to(map_)

                # Save map as HTML
                output_file = os.path.join(output_dir, f"journey_{journey_id}.html")
                map_.save(output_file)
                print(f"Map saved: {output_file}")
            
            except Exception as e:
                print(f"Error processing journey ID {journey_id}: {e}")

    def relative_ranges(self, base_column: str, output_file: Union[str, None] = None) -> pd.DataFrame:
        """
        Calculates statistical measures for each column relative to the base column in the DataFrame.
        """
        try:
            if base_column not in self.df.columns:
                raise ValueError(f"Base column '{base_column}' not found in DataFrame.")

            agg_columns = [col for col in self.df.columns if col != base_column]
            dfs = []

            for col in agg_columns:
                dff = self.df.groupby(base_column)[col].nunique().reset_index()

                column_ranges = {
                    'IQ_Range': '[' + str(round(dff[col].quantile(0.25), 1)) + '-' +
                                str(round(dff[col].quantile(0.75), 1)) + ':' +
                                str(round(dff[col].quantile(0.50), 1)) +
                                ']',
                    'Mean': str(round(dff[col].mean(), 1)) + '(' + str(round(dff[col].std(), 1)) + ')',
                    'MinMax': '(' + str(dff[col].min()) + ',' + str(dff[col].max()) + ')'
                }

                column_ranges_df = pd.DataFrame(column_ranges, index=[0])

                transposed_df = column_ranges_df.T

                dfs.append(transposed_df)

            combined_df = pd.concat(dfs, ignore_index=True, axis=1)
            combined_df.columns = agg_columns

            if output_file is not None:
                with open(output_file, 'w') as f:
                    f.write(f"<h1>Relative Measures to {base_column} </h1>\n")
                    f.write(combined_df.to_html(index=True))

            return combined_df
        
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def generate_correlation_heatmap(self, output_file: Union[str, None] = None, x_rotation: int = 90, y_rotation: int = 0) -> Union[str, None]:
        """
        Generates a refined correlation heatmap of numerical variables in the given pandas DataFrame.
        """
        try:
            numeric_df = self.df.select_dtypes(include=np.number)
            
            if numeric_df.empty:
                print("No numerical columns found in the DataFrame.")
                return None

            corr_matrix = numeric_df.corr()

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, linecolor='white')
            plt.title("Correlation Heatmap of Numerical Variables")
            plt.xticks(rotation=x_rotation)
            plt.yticks(rotation=y_rotation)

            if output_file:
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()
                return output_file
            else:
                plt.show()
                return None
        except Exception as e:
            print("An error occurred:", str(e))
            return None

    def generate_correlation_table(self, output_file: Union[str, None] = None) -> Union[pd.DataFrame, None]:
        """
        Generates a correlation table of numerical variables in the given DataFrame.
        """
        try:
            numeric_df = self.df.select_dtypes(include=np.number)
            
            if numeric_df.empty:
                return None
            
            corr_matrix = numeric_df.corr()

            corr_table_html = corr_matrix.to_html()

            if output_file:
                with open(output_file, "w") as f:
                    f.write(corr_table_html)
            
            return corr_matrix
        except Exception as e:
            logging.error("An error occurred: %s", str(e))
            return None

    def select_random_rows_without_missing_values(self) -> pd.DataFrame:
        """
        Selects random rows from a DataFrame, ensuring no missing values in any of the columns selected.

        For each column, the function selects 5 random rows without missing values, or all available rows if there are fewer than 5 non-missing.

        Returns:
            pandas.DataFrame: A DataFrame containing the selected random rows without missing values.
        """
        selected_rows = []

        # Iterate over each column
        for col in self.df.columns:
            # Filter out rows with missing values in the current column
            non_missing_rows = self.df[self.df[col].notnull()]

            # If there are fewer than 5 non-missing rows, include all of them
            if len(non_missing_rows) <= 5:
                selected_rows.extend(non_missing_rows.index)
            else:
                # Randomly select 5 rows without replacement
                selected_rows.extend(np.random.choice(non_missing_rows.index, size=5, replace=False))

        # Select the rows from the original DataFrame
        selected_df = self.df.loc[selected_rows]

        return selected_df

    def generate_html_stats(self, output_file: str) -> None:
        """
        Generates a descriptive statistics report for the DataFrame and writes it as an HTML file.

        Parameters:
            output_file (str): The path where the HTML file will be saved.

        Returns:
            None: The function writes the HTML file to the specified location.
        """
        stats_df = self.df.describe().transpose()
        stats_df['iQ'] = stats_df['75%'] - stats_df['25%']

        html = stats_df.to_html()

        # Write HTML to file
        with open(output_file, "w") as f:
            f.write(html)

    def summary_html(self, output_file: Union[str, None] = None) -> Union[pd.DataFrame, None]:
        """
        Generates a summary of numeric columns with descriptive statistics and interquartile range (iQ),
        and writes the result to an HTML file if an output file path is provided.

        Parameters:
            output_file (str or None): The path where the HTML file will be saved. If None, the result will not be saved.

        Returns:
            pandas.DataFrame: A DataFrame containing the summary statistics and iQ for each numeric column.
        """
        # Filter only numerical columns
        numeric_df = self.df.select_dtypes(include=np.number)

        if numeric_df.empty:
            return None

        # Calculate number of unique values for each column
        unique_counts = numeric_df.nunique()

        # Calculate descriptive statistics
        stats_df = numeric_df.describe().transpose()

        # Calculate interquartile range (iQ)
        stats_df['iQ'] = stats_df['75%'] - stats_df['25%']

        # Combine quartile and iQ values into a single column
        stats_df['Quartiles_iQ'] = stats_df.apply(lambda row: f"{row['25%']:.2f} - {row['75%']:.2f} ({row['iQ']:.2f})", axis=1)

        # Add column for number of unique values
        stats_df['Unique Values'] = unique_counts

        # Drop unnecessary columns
        stats_df.drop(columns=['25%', '75%', 'iQ'], inplace=True)

        html = stats_df.to_html()

        if output_file:
            # Write HTML to file
            with open(output_file, "w") as f:
                f.write(html)

        return stats_df

    def select_rows_with_missing_values(self, output_file: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Selects 5 rows with missing values for each column and generates a DataFrame.
        If `output_file` is provided, it writes the DataFrame to an HTML file.

        Parameters:
        output_file (str, optional): The path to the HTML file where the DataFrame will be saved.

        Returns:
        pandas.DataFrame: A DataFrame containing the first 5 rows with missing values for each column.
        None: If the `output_file` is provided, no return value is returned.
        """
        missing_rows = pd.DataFrame()

        # For each column, select the first 5 rows with missing values
        for column_name in self.df.columns:
            rows_with_missing_values = self.df[self.df[column_name].isnull()].head(5)
            missing_rows = pd.concat([missing_rows, rows_with_missing_values], ignore_index=True)

        if output_file:
            missing_rows.to_html(output_file, index=False)  # Save to HTML file if path is provided
        else:
            return missing_rows  # Return the DataFrame if no output_file is provided

    def convert_to_nearest_datatype(self) -> pd.DataFrame:
        """
        Converts columns to their appropriate datatypes based on the data content.

        - Converts columns containing only 'TRUE' or 'FALSE' to boolean.
        - Converts columns with numeric-like values to numeric data types.
        - Converts columns with 'date' or 'time' in their names to datetime or time.

        Returns:
        pandas.DataFrame: The DataFrame with converted column types.
        """
        for col in self.df.columns:
            # Convert to boolean if values are 'TRUE' or 'FALSE'
            if self.df[col].isin(['TRUE', 'FALSE']).all():
                self.df[col] = self.df[col].map({'TRUE': True, 'FALSE': False})
            # Convert to numeric if all values are numeric-like
            elif pd.to_numeric(self.df[col], errors='coerce').notna().all():
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            # Convert to datetime if column name contains 'date'
            elif 'date' in col:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            # Convert to time if column name contains 'time'
            elif 'time' in col:
                self.df[col] = pd.to_datetime(self.df[col], format='%H:%M:%S', errors='coerce').dt.time
        return self.df
    def show_column_types(self) -> None:
        """
        Prints the data types of each column in the DataFrame.
        """
        column_types_dict = {col: dtype for col, dtype in zip(self.df.columns, self.df.dtypes)}
        for column, data_type in column_types_dict.items():
            print(f"{column} : {data_type}")

    def add_missing_indicator_column(self, threshold: float = 0.8, colname: str = 'missing') -> pd.DataFrame:
        """
        Adds a new boolean column indicating whether a row has at least `threshold * 100%` missing values.
        """
        missing_values_per_row = self.df.isnull().sum(axis=1)
        total_columns = len(self.df.columns)
        missing_proportion_per_row = missing_values_per_row / total_columns
        self.df[colname] = missing_proportion_per_row >= threshold
        return self.df

    def null_rows_with_non_null_column(self, null_column: str, non_null_column: str) -> Optional[pd.DataFrame]:
        """
        Retrieves rows where one column has null values and another column has non-null values.
        """
        try:
            null_rows = self.df[self.df[null_column].isnull()]
            result_rows = null_rows[null_rows[non_null_column].notnull()]
            return result_rows
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def relative_non_null(self, base_column: str, output_file: Union[str, None] = None) -> pd.DataFrame:
        """
        Compute the count of null and non-null values for each column relative to the base column.
        """
        try:
            if base_column not in self.df.columns:
                raise ValueError(f"Base column '{base_column}' not found in DataFrame.")

            other_columns = [col for col in self.df.columns if col != base_column]
            null_counts = []
            non_null_counts = []
            unique_value_counts = []
            Unique_Values_Count_Reference = []

            base_null_count = self.df[base_column].isnull().sum()

            for col in other_columns:
                null_count = self.df[self.df[base_column].isnull()][col].isnull().sum()
                null_counts.append(null_count)

                non_null_count = self.df[self.df[base_column].isnull()][col].notnull().sum()
                non_null_percentage = (non_null_count / base_null_count) * 100
                non_null_counts.append(f"{non_null_count} ({non_null_percentage:.1f}%)")

                unique_value_count = self.df[self.df[base_column].isnull()][col].nunique()
                Unique_Values_Count_ref = self.df[col].nunique()
                Unique_Values_Count_Reference.append(Unique_Values_Count_ref)

                unique_value_ratio = (unique_value_count / Unique_Values_Count_ref) * 100
                unique_value_counts.append(f"{unique_value_count} ({unique_value_ratio:.1f}%)")

            combined_df = pd.DataFrame({'Features': other_columns, 
                                        'Null_Count': null_counts, 
                                        'Non_Null_Count': non_null_counts,
                                        'Unique_Values_Count': unique_value_counts,
                                        'Unique_Values_Count_Reference': Unique_Values_Count_Reference})

            if output_file is not None:
                with open(output_file, 'w') as f:
                    f.write("<html>\n<head>\n<title>Relative Null and Non-Null Counts</title>\n</head>\n<body>\n")
                    f.write(f"<h1>Relative Null and Non-Null Counts to Base Column: {base_column}</h1>\n")
                    f.write(f"<p>Number of null values in base column '{base_column}': {base_null_count}</p>\n")
                    f.write(combined_df.to_html(index=False, justify='center'))
                    f.write("\n</body>\n</html>")

            return combined_df

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return None

    def column_statistics(self, output_html: Union[str, None] = None) -> pd.DataFrame:
        """
        Calculate statistics for each column in the DataFrame.
        """
        try:
            # Start logging
            logging.basicConfig(filename=output_html, level=logging.INFO)
            logging.info("Starting column statistics calculation.")

            column_info = []
            total_rows = len(self.df)

            for col in self.df.columns:
                unique_values = self.df[col].nunique()
                missing_values = self.df[col].isnull().sum()
                total_values = total_rows
                proportion_unique = round(unique_values / total_values * 100, 2)

                column_info.append({
                    'Column': col,
                    'Unique Values': unique_values,
                    'Missing Values': missing_values,
                    'Total Values': total_values,
                    'Proportion Unique (%)': proportion_unique
                })

            stats_df = pd.DataFrame(column_info)

            # Sort the DataFrame based on the highest proportion of unique values to the total number of values
            stats_df = stats_df.sort_values(by='Proportion Unique (%)', ascending=False)

            if output_html:
                with open(output_html, 'w') as f:
                    f.write('<h1>Column Statistics Report</h1>')
                    f.write(stats_df.to_html(index=False))

            logging.info("Column statistics calculation completed successfully.")
            return stats_df

        except Exception as e:
            # Log error
            logging.error(f"An error occurred: {str(e)}")
            raise

    def read_files(self, data_directory: str) -> tuple:
        """
        Reads all JSON files from the given directory and combines them into a single DataFrame.
        """
        data_frames = []
        errors = defaultdict(list)
        loaded_files = []

        file_names = os.listdir(data_directory)

        for file_name in file_names:
            file_path = os.path.join(data_directory, file_name)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data)
                    data_frames.append(df)
                    loaded_files.append(file_name)
            except Exception as e:
                errors['error'].append(f"Error reading {file_path}: {str(e)}")

        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            return combined_df, errors, loaded_files
        else:
            return None, errors, loaded_files

    def remove_columns_rows_with_single_value_or_null(self, output_file: Optional[str] = None,
                                                      threshold_col: float = 0.5, threshold_row: float = 0.75) -> pd.DataFrame:
        """
        Removes columns with all null values or single unique values, and rows with excessive missing data.
        Optionally writes a summary to an HTML file.
        """
        # Count missing values per row and column
        missing_values_per_row = self.df.isnull().sum(axis=1)
        missing_values_per_col = self.df.isnull().sum(axis=0)

        # Remove columns with all null values or with too many missing values
        null_columns = self.df.columns[missing_values_per_col / len(self.df) >= threshold_col]
        self.df = self.df.drop(columns=null_columns)

        # Remove columns with only one unique value
        single_value_columns = [col for col in self.df.columns if self.df[col].nunique() == 1]
        self.df = self.df.drop(columns=single_value_columns)

        # Remove rows with excessive missing values
        threshold_null = threshold_row * self.df.shape[1]
        self.df = self.df.dropna(axis=0, thresh=threshold_null)

        # Remove rows with all null values
        self.df = self.df.dropna(axis=0, how='all')

        # Final number of rows and columns after cleaning
        cleaned_num_rows, cleaned_num_cols = self.df.shape

        if output_file:
            # Write summary statistics to HTML file if requested
            with open(output_file, 'w') as f:
                f.write("<h1>Data Cleaning Summary</h1>")
                f.write(f"Original DataFrame had {self.df.shape[0]} rows and {self.df.shape[1]} columns.<br>")
                f.write(f"Cleaned DataFrame has {cleaned_num_rows} rows and {cleaned_num_cols} columns.<br>")
                f.write("<h2>Removed Columns</h2>")
                f.write(f"Columns removed due to all-null values: {', '.join(null_columns)}<br>")
                f.write(f"Columns removed with single unique values: {', '.join(single_value_columns)}<br>")
                f.write("<h2>Remaining Data</h2>")
                f.write(f"Remaining columns: {', '.join(self.df.columns)}<br>")

        return self.df

