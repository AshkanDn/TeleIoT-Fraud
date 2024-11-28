

"""
Enrichment Module for GeoSpatial and Sensor Data Analysis
====================================
This module provides advanced data enrichment techniques for sensor and geospatial datasets, focusing on trip metrics, signal strength, wavelet transformations, and principal component analysis (PCA). 
The functions within this module are designed to process time-series data and compute a variety of derived features that are essential for enhanced analysis, predictive modeling, and decision-making in geospatial and sensor-based applications. 

@Lead Analyst: Ashkan Dashtban

Key Functionalities:
- **Geospatial Analysis**: Calculation of great-circle distance, direct distance, and trip metrics (e.g., total distance, average speed, duration).
- **Signal Strength Metrics**: Computation of signal strength based on satellite count and Horizontal Dilution of Precision (HDOP).
- **Wavelet Analysis**: Extraction of key wavelet components from accelerometer data (XYZ axes) and geospatial coordinates (latitude, longitude).
- **Principal Component Analysis (PCA)**: Dimensionality reduction and feature extraction from complex datasets, enhancing the interpretability of the data.
- **Distinct Feature Counting**: Counting the number of unique values in a given feature, enriching the dataset for more in-depth analysis.

Functions:

1. **haversine(lat1, lon1, lat2, lon2)**:
   - Calculates the great-circle distance (in kilometers) between two points on the Earth's surface using the Haversine formula. This function is critical for measuring distances between geospatial coordinates.

2. **add_trip_metrics(df, id_col='ID', lat_col='coordinates.lat', lon_col='coordinates.lon', timestamp_col='ts')**:
   - Computes and enriches the DataFrame with trip metrics for each unique trip ID, including total trip distance, average speed, trip duration, and the straight-line distance between the first and last coordinates.

3. **add_gforce(df, averageX='averageX', averageY='averageY', averageZ='averageZ')**:
   - Calculates the g-force based on the average X, Y, and Z accelerations, contributing valuable insight into the intensity of movements during each trip.

4. **compute_signal_strength(num_sats_series, hdop_series)**:
   - Computes a normalized signal strength score using the number of satellites in view and Horizontal Dilution of Precision (HDOP). This metric is essential for understanding the quality of GPS signals during data collection.

5. **add_signal_strength(df, id_col='ID', num_sats_col='numSats', hdop_col='hdop', signal_strength_col='Signal_Strength')**:
   - Enriches the DataFrame by adding a signal strength column, calculated from 'numSats' (satellite count) and 'hdop' (HDOP), for each unique trip ID.

6. **add_wavelet_xyz(df, id_col='ID', ncomp=4, avgx='averageX', avgy='averageY', avgz='averageZ')**:
   - Extracts the highest-energy wavelet components from accelerometer data (XYZ axes) for each trip, providing key insights into the underlying motion dynamics.

7. **add_wavelet_lonlat(df, id_col='ID', ncomp=4, lat_col='coordinates.lat', long_col='coordinates.lon')**:
   - Extracts wavelet components from latitude and longitude series for each trip and adds the top `ncomp` wavelet components as new features, improving the granularity of geospatial analysis.

8. **add_pca(df, feature_col, ncomp=2, prefix='PCA_', id_col='ID')**:
   - Performs Principal Component Analysis (PCA) on a specified feature column per trip ID, reducing the dimensionality and adding the top `ncomp` principal components as new columns for enhanced analysis.

9. **add_distinct_count(df, feature_col='feature', id_col='ID')**:
   - Computes the count of distinct values for a specified feature, enriching the dataset with insights about the variety of data points within each trip.

10. **enrichment_pipeline(df, id_col='UID')**:
    - A comprehensive, high-level function that orchestrates the execution of all enrichment functions in a sequential manner. This pipeline enriches the DataFrame with trip metrics, signal strength, wavelet components, PCA components, and distinct value counts for various features.

Important Notes:
- Ensure that the input DataFrame contains the necessary columns (e.g., coordinates, timestamps, accelerometer data) for each function to operate correctly.
- The input data should be pre-processed and cleaned to handle missing or erroneous values.

Note: All functions are intended to be used within the `enrichment_pipeline` function.

"""


# Standard Library Imports
import os
import json
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import pywt
from sklearn.decomposition import PCA
from typing import Union

# Logging Configuration
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface using the Haversine formula.

    Parameters:
        lat1, lon1: Coordinates of the first point (degrees).
        lat2, lon2: Coordinates of the second point (degrees).

    Returns:
        distance (float): Distance in kilometers between the two points.
    """
    try:
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = 6371 * c  # Radius of the Earth in kilometers
        
        return distance
    except Exception as e:
        logging.error(f"Error in haversine calculation: {e}")
        return None

def add_trip_metrics(df: pd.DataFrame, id_col: str = 'ID', 
                     lat_col: str = "coordinates.lat", lon_col: str = "coordinates.lon", 
                     timestamp_col: str = "ts") -> pd.DataFrame:
    """
    Compute trip metrics such as total distance, average speed, duration, 
    and direct distance between the first and last points for each ID.

    Parameters:
        df: Input DataFrame containing GeoSpatial data.
        id_col: Column representing unique trip IDs.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        timestamp_col: Timestamp column name.

    Returns:
        df: DataFrame with new trip metrics columns added.
    """
    
    # Initialize metric columns
    df['total_distance'] = 0.0
    df['average_speed'] = 0.0
    df['duration_minutes'] = 0.0
    df['direct_distance'] = 0.0

    # Iterate over each unique ID group
    for trip_id, group in df.groupby(id_col):
        # Extract necessary data as lists
        latitudes = group[lat_col].to_list()
        longitudes = group[lon_col].to_list()
        timestamps = group[timestamp_col].to_list()

        # Initialize accumulators
        total_distance = 0.0
        total_duration = 0.0
        prev_lat, prev_lon, prev_time = None, None, None

        # Calculate total distance and duration
        for lat, lon, ts in zip(latitudes, longitudes, timestamps):
            if None in (lat, lon, ts):
                continue  # Skip incomplete data points

            if prev_lat is not None and prev_lon is not None:
                # Increment distance
                total_distance += haversine(prev_lat, prev_lon, lat, lon)

                # Increment duration in seconds
                time_diff = (datetime.fromisoformat(ts) - datetime.fromisoformat(prev_time)).total_seconds()
                total_duration += time_diff

            # Update previous values
            prev_lat, prev_lon, prev_time = lat, lon, ts

        # Calculate average speed (km/h)
        average_speed = (total_distance / (total_duration / 3600)) if total_duration > 0 else 0.0

        # Calculate duration in minutes
        duration_minutes = total_duration / 60

        # Calculate direct distance (km)
        if latitudes and longitudes:
            direct_distance = haversine(latitudes[0], longitudes[0], latitudes[-1], longitudes[-1])
        else:
            direct_distance = 0.0

        # Update metrics for the current trip
        df.loc[df[id_col] == trip_id, 'total_distance'] = total_distance
        df.loc[df[id_col] == trip_id, 'average_speed'] = average_speed
        df.loc[df[id_col] == trip_id, 'duration_minutes'] = duration_minutes
        df.loc[df[id_col] == trip_id, 'direct_distance'] = direct_distance

    return df

def add_gforce(df: pd.DataFrame, averageX: str = "averageX", averageY: str = "averageY", averageZ: str = "averageZ") -> pd.DataFrame:
    """
    Compute the g-force based on average acceleration components and add a new column 'gforce' to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing average acceleration components.
        averageX (str): the X-axis acceleration component.
        averageY (str): the Y-axis acceleration component.
        averageZ (str): the Z-axis acceleration component.

    Returns:
        pd.DataFrame: DataFrame with a new column 'gforce' added.
    """
    # Define a function to compute g-force for a single row
    #def gforce_for_row(row):
    #    magnitude = np.sqrt(row[averageX]**2 + row[averageY]**2 + row[averageZ]**2)
    #    gforce = magnitude / 9.81
    #    return gforce
    # Apply the function to each row in the DataFrame
    #df['gforce'] = df.apply(gforce_for_row, axis=1)
    
    # Compute g-force magnitude (optimised version of the above code)
    df['gforce'] = np.sqrt(df[averageX]**2 + df[averageY]**2 + df[averageZ]**2) / 9.81
    return df

def compute_signal_strength(num_sats_series: pd.Series, hdop_series: pd.Series) -> pd.Series:
    """
    Compute signal strength based on the number of satellites and HDOP (Horizontal Dilution of Precision).

    Parameters:
        num_sats_series (pd.Series): Series containing the number of satellites.
        hdop_series (pd.Series): Series containing HDOP values.

    Returns:
        pd.Series: Series containing computed signal strength values normalized to the range [0, 1].
    """
    try:
        # Normalize numSats and HDOP values to the range [0, 1]
        normalized_num_sats = (num_sats_series - num_sats_series.min()) / (num_sats_series.max() - num_sats_series.min())
        normalized_hdop = (hdop_series.max() - hdop_series) / (hdop_series.max() - hdop_series.min())

        # Compute the weighted average of normalized values
        signal_strength = (normalized_num_sats + normalized_hdop) / 2

        return signal_strength
    except ZeroDivisionError:
        print("Error: Division by zero occurred while normalizing values. Check input data.")
        return pd.Series([None] * len(num_sats_series))
    except Exception as e:
        print(f"Unexpected error in compute_signal_strength: {e}")
        return pd.Series([None] * len(num_sats_series))

def add_signal_strength(df: pd.DataFrame, 
                        id_col: str = 'ID', 
                        num_sats_col: str = 'numSats', 
                        hdop_col: str = 'hdop', 
                        signal_strength_col: str = 'Signal_Strength') -> pd.DataFrame:
    """
    Compute signal strength based on 'numSats' and 'hdop' for each unique ID and add it as a new column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        id_col (str): Column name representing unique IDs for grouping.
        num_sats_col (str): Column name for the number of satellites.
        hdop_col (str): Column name for HDOP (Horizontal Dilution of Precision).
        signal_strength_col (str): Name of the new column to store computed signal strength.

    Returns:
        pd.DataFrame: DataFrame with the computed signal strength added as a new column.
    """
    try:
        # Verify that required columns exist in the DataFrame
        required_cols = {id_col, num_sats_col, hdop_col}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        # Define a helper function to compute signal strength for a group
        def compute_group_signal_strength(group):
            return compute_signal_strength(group[num_sats_col], group[hdop_col])

        # Apply computation grouped by ID
        signal_strength = df.groupby(id_col).apply(compute_group_signal_strength)

        # Assign computed signal strength back to the original DataFrame
        df[signal_strength_col] = signal_strength.reset_index(level=0, drop=True)

        return df

    except Exception as e:
        print(f"Error in add_signal_strength: {e}")
        return df.copy()  # Return a copy of the original DataFrame in case of an error

def add_wavelet_xyz(df: pd.DataFrame, 
                    id_col: str = 'ID', 
                    ncomp: int = 4, 
                    avgx: str = 'averageX', 
                    avgy: str = 'averageY', 
                    avgz: str = 'averageZ') -> pd.DataFrame:
    """
    Extract the wavelet components with the highest energy from average X, Y, and Z series per ID.
    Adds the top `ncomp` components as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        id_col (str): Column name representing unique IDs for grouping.
        ncomp (int): Number of wavelet components with the highest energy to extract.
        avgx (str): Column name for the X-axis average.
        avgy (str): Column name for the Y-axis average.
        avgz (str): Column name for the Z-axis average.

    Returns:
        pd.DataFrame: Modified DataFrame with added wavelet components.
    """
    try:
        # Check if required columns exist
        required_cols = {id_col, avgx, avgy, avgz}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        # Function to compute wavelet components for each group
        def compute_wavelet_components(group):
            try:
                # Combine X, Y, Z into a trajectory
                trajectory = np.vstack((group[avgx].to_numpy(), 
                                        group[avgy].to_numpy(), 
                                        group[avgz].to_numpy()))

                # Perform Discrete Wavelet Transform (DWT)
                coeffs = pywt.dwt2(trajectory, 'haar')

                # Calculate energy of wavelet components
                components = coeffs[0]
                energies = [np.sum(np.abs(component) ** 2) for component in components]

                # Sort by energy and select top `ncomp` components
                sorted_indices = np.argsort(energies)[::-1][:ncomp]
                selected_components = [components[i] for i in sorted_indices]

                # Flatten components into a dictionary for output
                result = {f'Wavelet_XYZ_{i+1}': selected_components[i] for i in range(len(selected_components))}
                return pd.Series(result)

            except Exception as e:
                print(f"Error in compute_wavelet_components for group {group[id_col].iloc[0]}: {e}")
                return pd.Series({f'Wavelet_XYZ_{i+1}': np.nan for i in range(ncomp)})

        # Apply the computation to each group
        wavelet_results = df.groupby(id_col).apply(compute_wavelet_components)

        # Reset index and merge the results back to the original DataFrame
        wavelet_results = wavelet_results.reset_index()
        df = df.merge(wavelet_results, on=id_col, how='left')

        return df

    except Exception as e:
        print(f"Error occurred in add_wavelet_xyz: {e}")
        return df.copy()

def add_wavelet_lonlat(df: pd.DataFrame, 
                        id_col: str = 'ID', 
                        ncomp: int = 4, 
                        lat_col: str = 'coordinates.lat', 
                        long_col: str = 'coordinates.lon') -> pd.DataFrame:
    """
    Extract the wavelet components with the highest energy from the latitude and longitude series per ID.
    Adds the top `ncomp` wavelet components as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        id_col (str): Column name representing unique IDs for grouping.
        ncomp (int): Number of wavelet components with the highest energy to extract.
        lat_col (str): Column name for the latitude.
        long_col (str): Column name for the longitude.

    Returns:
        pd.DataFrame: Modified DataFrame with added wavelet components.
    """
    try:
        # Check if required columns exist in the dataframe
        required_cols = {id_col, lat_col, long_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Function to compute wavelet components for each group
        def compute_wavelet_components(group):
            try:
                # Concatenate latitude and longitude into a trajectory
                trajectory = np.vstack((group[lat_col].to_numpy(), group[long_col].to_numpy()))

                # Perform Discrete Wavelet Transform (DWT)
                coeffs = pywt.dwt2(trajectory, 'haar')

                # Compute energy of each wavelet component
                components = coeffs[0]
                energies = [np.sum(np.abs(component) ** 2) for component in components]

                # Sort by energy and select top `ncomp` components
                sorted_indices = np.argsort(energies)[::-1][:ncomp]
                selected_components = [components[i] for i in sorted_indices]

                # Flatten components into a dictionary for output
                result = {f'Wavelet_LonLat_{i+1}': selected_components[i] for i in range(len(selected_components))}
                return pd.Series(result)

            except Exception as e:
                print(f"Error in compute_wavelet_components for group {group[id_col].iloc[0]}: {e}")
                return pd.Series({f'Wavelet_LonLat_{i+1}': np.nan for i in range(ncomp)})

        # Apply the computation to each group
        wavelet_results = df.groupby(id_col).apply(compute_wavelet_components)

        # Merge the wavelet results back to the original DataFrame
        df = df.merge(wavelet_results.reset_index(), on=id_col, how='left')

        return df

    except Exception as e:
        print(f"Error occurred in add_wavelet_lonlat: {e}")
        return df.copy()

def add_pca(df: pd.DataFrame, feature_col: str, ncomp: int = 2, prefix: str = 'PCA_', id_col: str = 'ID') -> pd.DataFrame:
    """
    Extract principal components from the specified feature column per ID and add them to the dataframe.
    This function applies PCA to each group of data identified by the `id_col` and adds the top `ncomp` principal components
    as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        feature_col (str): Column name containing the feature values to apply PCA on.
        ncomp (int): Number of principal components to extract.
        prefix (str): Prefix for the newly created PCA columns.
        id_col (str): Column name representing the unique identifiers for grouping.

    Returns:
        pd.DataFrame: DataFrame with new PCA components added as columns.
    """
    try:
        # Initialize PCA object
        pca = PCA(n_components=ncomp)

        # Check if the required columns exist in the DataFrame
        if id_col not in df.columns or feature_col not in df.columns:
            missing_cols = [col for col in [id_col, feature_col] if col not in df.columns]
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

        # Initialize empty lists to store the PCA results
        pca_components = []

        # Iterate over each group identified by `id_col`
        for group_id, group_df in df.groupby(id_col):
            # Extract feature values for PCA
            column_values = group_df[feature_col].values.reshape(-1, 1)

            # Ensure that there are enough data points to perform PCA
            if len(column_values) >= ncomp:
                # Fit and transform PCA
                pca_result = pca.fit_transform(column_values)
            else:
                # If not enough data, fill the result with NaN values
                pca_result = np.full((len(column_values), ncomp), np.nan)

            # Append the PCA results for this group
            pca_components.append(pca_result)

        # Concatenate all PCA components and merge them back to the original dataframe
        pca_components = np.vstack(pca_components)  # Combine the results across all groups
        for i in range(ncomp):
            df[f'{prefix}{i+1}'] = pca_components[:, i]

        return df

    except Exception as e:
        print(f"Error occurred in add_pca: {e}")
        return df.copy()  # Return the original dataframe in case of an error

def add_distinct_count(df: pd.DataFrame, feature_col: str = 'feature', id_col: str = 'ID') -> pd.DataFrame:
    """
    Compute the number of distinct values of a feature per ID and add it to the dataframe as a new column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        feature_col (str): The name of the column containing the feature whose distinct values need to be counted.
        id_col (str): The column used for grouping (typically ID column).
        
    Returns:
        pd.DataFrame: The original dataframe with an additional column representing the count of distinct feature values per ID.
    """
    try:
        # Ensure the feature column exists in the dataframe
        if feature_col not in df.columns:
            raise ValueError(f"'{feature_col}' column not found in the DataFrame.")

        # Count distinct feature values per ID
        distinct_counts = df.groupby(id_col)[feature_col].nunique()

        # Check if there are any valid distinct counts
        if distinct_counts.empty:
            raise ValueError(f"No distinct values found for '{feature_col}' column.")

        # Add the distinct count as a new column in the dataframe
        df['Distinct_' + feature_col + '_Count'] = df[id_col].map(distinct_counts)

        return df
    except Exception as e:
        print(f"Error occurred: {e}")
        return df.copy()  # Return a copy of the original dataframe in case of an error

def enrichment_pipeline(df: pd.DataFrame, id_col: str = "UID") -> pd.DataFrame:
    """
    Enrich the input DataFrame by applying various feature extraction functions in sequence.
    Each function adds new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing raw data.
        id_col (str): The column used to group data by ID (default is "UID").
    
    Returns:
        pd.DataFrame: The enriched DataFrame with new computed columns.
    """
    try:
        # Apply trip metrics enrichment
        df = add_trip_metrics(df, id_col=id_col)
        
        # Apply gforce calculation
        df = add_gforce(df)
        
        # Apply signal strength calculation
        df = add_signal_strength(df, id_col=id_col)
        
        # Apply wavelet transformations for XYZ components
        df = add_wavelet_xyz(df, ncomp=3, id_col=id_col)
        
        # Apply wavelet transformations for LonLat components
        df = add_wavelet_lonlat(df, ncomp=3, id_col=id_col)
        
        # Apply PCA on speed (in mph)
        df = add_pca(df, ncomp=1, feature_col="speedMph", prefix="PCA_speedMph_", id_col=id_col)
        
        # Apply PCA on gforce
        df = add_pca(df, ncomp=1, feature_col="gforce", prefix="PCA_gforce_", id_col=id_col)
        
        # Apply distinct count for formOfWay feature
        df = add_distinct_count(df, feature_col='formOfWay', id_col=id_col)
        
        return df
    
    except Exception as e:
        print(f"An error occurred during enrichment: {e}")
        return df.copy()  # Return a copy of the original DataFrame in case of error



























