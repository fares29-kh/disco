"""
Metrics Calculation Module for Process Mining Dashboard
"""

import pandas as pd
import numpy as np
from datetime import timedelta


def calculate_case_durations(df):
    """
    Calculate duration for each case (bug).
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: DataFrame with case_id, start_time, end_time, duration_hours
    """
    case_summary = df.groupby('case_id').agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    
    case_summary.columns = ['case_id', 'start_time', 'end_time']
    case_summary['duration'] = case_summary['end_time'] - case_summary['start_time']
    case_summary['duration_hours'] = case_summary['duration'].dt.total_seconds() / 3600
    
    return case_summary


def calculate_kpis(df, sla_threshold):
    """
    Calculate key performance indicators.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        
    Returns:
        dict: Dictionary containing all KPIs
    """
    kpis = {}
    
    # Total bugs
    kpis['total_bugs'] = df['case_id'].nunique()
    
    # Date range
    kpis['start_date'] = df['timestamp'].min()
    kpis['end_date'] = df['timestamp'].max()
    kpis['period_days'] = (kpis['end_date'] - kpis['start_date']).days
    
    # Calculate case durations
    case_durations = calculate_case_durations(df)
    
    # Slowest bug
    slowest_case = case_durations.loc[case_durations['duration_hours'].idxmax()]
    kpis['slowest_bug_id'] = slowest_case['case_id']
    kpis['slowest_bug_duration'] = slowest_case['duration_hours']
    
    # Average resolution time
    kpis['avg_resolution_time'] = case_durations['duration_hours'].mean()
    
    # SLA at-risk percentage
    at_risk_cases = case_durations[case_durations['duration_hours'] > sla_threshold]
    kpis['sla_risk_percentage'] = (len(at_risk_cases) / len(case_durations)) * 100 if len(case_durations) > 0 else 0
    kpis['sla_risk_count'] = len(at_risk_cases)
    
    # Number of reopenings
    reopen_count = df[df['activity'].str.lower().str.contains('reopen', na=False)]['case_id'].nunique()
    kpis['reopen_count'] = reopen_count
    kpis['reopen_rate'] = (reopen_count / kpis['total_bugs']) * 100 if kpis['total_bugs'] > 0 else 0
    
    # Completion rate (cases with "Close" activity)
    closed_cases = df[df['activity'].str.lower().str.contains('close', na=False)]['case_id'].nunique()
    kpis['completion_rate'] = (closed_cases / kpis['total_bugs']) * 100 if kpis['total_bugs'] > 0 else 0
    kpis['closed_count'] = closed_cases
    
    # Activity distribution
    kpis['activity_counts'] = df['activity'].value_counts().to_dict()
    
    # Category distribution (only if category column exists)
    if 'category' in df.columns:
        kpis['category_counts'] = df.groupby('case_id')['category'].first().value_counts().to_dict()
    else:
        kpis['category_counts'] = {}
    
    # Priority distribution (only if priority column exists)
    if 'priority' in df.columns:
        kpis['priority_counts'] = df.groupby('case_id')['priority'].first().value_counts().to_dict()
    else:
        kpis['priority_counts'] = {}
    
    return kpis


def calculate_activity_durations(df):
    """
    Calculate time spent in each activity for each case.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: DataFrame with case_id, activity, duration_hours
    """
    df_sorted = df.sort_values(['case_id', 'timestamp'])
    
    # Calculate time to next activity
    df_sorted['next_timestamp'] = df_sorted.groupby('case_id')['timestamp'].shift(-1)
    df_sorted['activity_duration'] = df_sorted['next_timestamp'] - df_sorted['timestamp']
    df_sorted['activity_duration_hours'] = df_sorted['activity_duration'].dt.total_seconds() / 3600
    
    # Remove last activity of each case (no duration)
    activity_durations = df_sorted[df_sorted['activity_duration'].notna()].copy()
    
    return activity_durations


def calculate_transition_durations(df):
    """
    Calculate average duration between activity transitions.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: DataFrame with from_activity, to_activity, avg_duration_hours, frequency
    """
    df_sorted = df.sort_values(['case_id', 'timestamp'])
    
    # Get next activity for each row
    df_sorted['next_activity'] = df_sorted.groupby('case_id')['activity'].shift(-1)
    df_sorted['next_timestamp'] = df_sorted.groupby('case_id')['timestamp'].shift(-1)
    
    # Calculate transition duration
    df_sorted['transition_duration'] = df_sorted['next_timestamp'] - df_sorted['timestamp']
    df_sorted['transition_duration_hours'] = df_sorted['transition_duration'].dt.total_seconds() / 3600
    
    # Remove last activity of each case
    transitions = df_sorted[df_sorted['next_activity'].notna()].copy()
    
    # Group by transition
    transition_summary = transitions.groupby(['activity', 'next_activity']).agg({
        'transition_duration_hours': ['mean', 'median', 'std', 'count']
    }).reset_index()
    
    transition_summary.columns = ['from_activity', 'to_activity', 'avg_duration_hours', 
                                   'median_duration_hours', 'std_duration_hours', 'frequency']
    
    transition_summary = transition_summary.sort_values('avg_duration_hours', ascending=False)
    
    return transition_summary


def identify_bottlenecks(df, sla_threshold, top_n=10):
    """
    Identify bottlenecks in the process.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        top_n: Number of top bottlenecks to return
        
    Returns:
        pd.DataFrame: DataFrame with bottleneck information
    """
    transitions = calculate_transition_durations(df)
    
    # Add SLA status
    def get_sla_status(duration):
        if duration > sla_threshold:
            return '❌ Critical'
        elif duration > sla_threshold * 0.5:
            return '⚠️ Warning'
        else:
            return '✅ OK'
    
    transitions['sla_status'] = transitions['avg_duration_hours'].apply(get_sla_status)
    
    # Format duration
    transitions['avg_duration_formatted'] = transitions['avg_duration_hours'].apply(
        lambda x: f"{x:.2f}h"
    )
    
    # Select top bottlenecks
    bottlenecks = transitions.head(top_n)
    
    return bottlenecks


def calculate_heatmap_data(df):
    """
    Calculate average duration by activity and category for heatmap.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Pivot table with activities as columns and categories as rows
    """
    # Check if required columns exist
    if 'category' not in df.columns:
        # Return empty DataFrame if category column doesn't exist
        return pd.DataFrame()
    
    # Calculate activity durations
    activity_durations = calculate_activity_durations(df)
    
    # Check if we have duration data
    if activity_durations.empty:
        return pd.DataFrame()
    
    # Get columns that exist in df
    cols_to_get = ['category']
    if 'priority' in df.columns:
        cols_to_get.append('priority')
    if 'severity' in df.columns:
        cols_to_get.append('severity')
    
    # Merge with case information to get category
    case_info = df.groupby('case_id')[cols_to_get].first().reset_index()
    activity_durations = activity_durations.merge(case_info, on='case_id', how='left')
    
    # Check if we have the necessary data after merge
    if 'category' not in activity_durations.columns or 'activity' not in activity_durations.columns:
        return pd.DataFrame()
    
    # Remove rows with null category or activity
    activity_durations = activity_durations.dropna(subset=['category', 'activity', 'activity_duration_hours'])
    
    if activity_durations.empty:
        return pd.DataFrame()
    
    # Create pivot table
    try:
        heatmap_data = activity_durations.pivot_table(
            values='activity_duration_hours',
            index='category',
            columns='activity',
            aggfunc='mean',
            fill_value=0
        )
        return heatmap_data
    except Exception as e:
        # If pivot fails, return empty DataFrame
        return pd.DataFrame()


def calculate_variant_analysis(df):
    """
    Analyze process variants (paths through the process).
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: DataFrame with variant paths and their frequencies
    """
    # Group activities by case_id to create process paths
    variants = df.groupby('case_id')['activity'].apply(lambda x: ' → '.join(x)).reset_index()
    variants.columns = ['case_id', 'path']
    
    # Count variant frequencies
    variant_counts = variants['path'].value_counts().reset_index()
    variant_counts.columns = ['path', 'frequency']
    variant_counts['percentage'] = (variant_counts['frequency'] / len(variants)) * 100
    
    return variant_counts


def calculate_category_statistics(df):
    """
    Calculate statistics by category.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Statistics by category
    """
    # Check if category column exists
    if 'category' not in df.columns:
        return pd.DataFrame(columns=['category', 'avg_duration', 'median_duration', 
                                    'std_duration', 'min_duration', 'max_duration', 'case_count'])
    
    case_durations = calculate_case_durations(df)
    
    # Get columns that exist in df
    cols_to_get = ['category']
    if 'priority' in df.columns:
        cols_to_get.append('priority')
    if 'severity' in df.columns:
        cols_to_get.append('severity')
    
    case_info = df.groupby('case_id')[cols_to_get].first().reset_index()
    case_durations = case_durations.merge(case_info, on='case_id', how='left')
    
    # Check if category column exists after merge
    if 'category' not in case_durations.columns:
        return pd.DataFrame(columns=['category', 'avg_duration', 'median_duration', 
                                    'std_duration', 'min_duration', 'max_duration', 'case_count'])
    
    # Remove null categories
    case_durations = case_durations.dropna(subset=['category'])
    
    if case_durations.empty:
        return pd.DataFrame(columns=['category', 'avg_duration', 'median_duration', 
                                    'std_duration', 'min_duration', 'max_duration', 'case_count'])
    
    try:
        category_stats = case_durations.groupby('category').agg({
            'duration_hours': ['mean', 'median', 'std', 'min', 'max', 'count']
        }).reset_index()
        
        category_stats.columns = ['category', 'avg_duration', 'median_duration', 
                                  'std_duration', 'min_duration', 'max_duration', 'case_count']
        
        category_stats = category_stats.sort_values('avg_duration', ascending=False)
        
        return category_stats
    except Exception as e:
        return pd.DataFrame(columns=['category', 'avg_duration', 'median_duration', 
                                    'std_duration', 'min_duration', 'max_duration', 'case_count'])

