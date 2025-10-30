"""
Data Loading and Validation Module for Process Mining Dashboard
"""

import pandas as pd
import streamlit as st
from datetime import datetime
import pytz


def standardize_column_names(df, column_mapping):
    """
    Standardize column names by mapping various formats to standard names.
    
    Args:
        df: Input dataframe
        column_mapping: Dictionary mapping standard names to list of possible names
        
    Returns:
        pd.DataFrame: Dataframe with standardized column names
    """
    df_renamed = df.copy()
    
    # Create reverse mapping (all possible names → standard name)
    reverse_mapping = {}
    for standard_name, possible_names in column_mapping.items():
        for possible_name in possible_names:
            reverse_mapping[possible_name] = standard_name
    
    # Rename columns that match
    columns_to_rename = {}
    for col in df_renamed.columns:
        if col in reverse_mapping:
            columns_to_rename[col] = reverse_mapping[col]
    
    if columns_to_rename:
        df_renamed = df_renamed.rename(columns=columns_to_rename)
        st.info(f"ℹ️ Renamed columns: {', '.join([f'{k}→{v}' for k, v in columns_to_rename.items()])}")
    
    return df_renamed


def load_and_validate_csv(uploaded_file):
    """
    Load and validate the event log file (CSV or Excel).
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        pd.DataFrame: Validated and processed dataframe
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    try:
        # Read file based on extension
        file_name = uploaded_file.name.lower()
        
        if file_name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            raise ValueError(f"Unsupported file format. Please upload CSV or Excel (.xlsx) file.")
        
        # Column mapping: support both old and new formats
        column_mapping = {
            # Old format → New format
            'case_id': ['case_id', 'case:concept:name'],
            'activity': ['activity', 'concept:name'],
            'timestamp': ['timestamp', 'time:timestamp'],
            'category': ['category', 'Category'],
            'priority': ['priority', 'Priority'],
            'severity': ['severity', 'Severity'],
            'main_impact': ['main_impact', 'main impact']
        }
        
        # Rename columns to standard format
        df = standardize_column_names(df, column_mapping)
        
        # Define required columns (using standardized names)
        required_columns = ['case_id', 'activity', 'timestamp', 'category', 'priority', 'severity']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Show user-friendly message with possible column names
            alt_names = []
            for col in missing_columns:
                if col in column_mapping:
                    alt_names.append(f"{col} (or {', '.join(column_mapping[col])})")
                else:
                    alt_names.append(col)
            raise ValueError(f"Missing required columns: {', '.join(alt_names)}")
        
        # Convert timestamp to datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except Exception as e:
            raise ValueError(f"Invalid timestamp format. Please use ISO format or standard datetime: {str(e)}")
        
        # Validate case_id (should not be null)
        if df['case_id'].isnull().any():
            st.warning(f"Found {df['case_id'].isnull().sum()} rows with null case_id. These rows will be removed.")
            df = df.dropna(subset=['case_id'])
        
        # Validate activity (should not be null)
        if df['activity'].isnull().any():
            st.warning(f"Found {df['activity'].isnull().sum()} rows with null activity. These rows will be removed.")
            df = df.dropna(subset=['activity'])
        
        # Handle missing values in optional columns
        df['category'] = df['category'].fillna('Unknown')
        df['priority'] = df['priority'].fillna('Medium')
        df['severity'] = df['severity'].fillna('Minor')
        
        # Sort by case_id and timestamp
        df = df.sort_values(['case_id', 'timestamp']).reset_index(drop=True)
        
        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        
        return df
        
    except pd.errors.EmptyDataError:
        raise ValueError("The uploaded CSV file is empty.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")
    except Exception as e:
        raise ValueError(f"Unexpected error loading data: {str(e)}")


def apply_filters(df, categories=None, priorities=None, severities=None, date_range=None):
    """
    Apply filters to the dataframe.
    
    Args:
        df: Input dataframe
        categories: List of categories to filter
        priorities: List of priorities to filter
        severities: List of severities to filter
        date_range: Tuple of (start_date, end_date)
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    filtered_df = df.copy()
    
    # Apply category filter
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    
    # Apply priority filter
    if priorities:
        filtered_df = filtered_df[filtered_df['priority'].isin(priorities)]
    
    # Apply severity filter
    if severities:
        filtered_df = filtered_df[filtered_df['severity'].isin(severities)]
    
    # Apply date range filter
    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) &
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    return filtered_df


def get_filter_options(df):
    """
    Get unique values for filter options.
    
    Args:
        df: Input dataframe
        
    Returns:
        dict: Dictionary containing unique values for each filter
    """
    return {
        'categories': sorted(df['category'].unique().tolist()),
        'priorities': sorted(df['priority'].unique().tolist()),
        'severities': sorted(df['severity'].unique().tolist()),
        'min_date': df['timestamp'].min().date(),
        'max_date': df['timestamp'].max().date()
    }


def convert_to_pm4py_log(df):
    """
    Convert dataframe to pm4py event log format.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Formatted event log for pm4py
    """
    # Rename columns to pm4py standard format
    log_df = df.copy()
    log_df = log_df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    
    # Sort by case and timestamp
    log_df = log_df.sort_values(['case:concept:name', 'time:timestamp'])
    
    return log_df

