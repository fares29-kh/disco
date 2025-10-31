"""
Feature Engineering Module for ML Predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime


def extract_features_from_log(df):
    """
    Extract features from event log for ML training.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Features dataframe with target variable (duration_hours)
    """
    if df.empty:
        return pd.DataFrame()
    
    # Calculate case durations
    case_durations = df.groupby('case_id').agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    case_durations.columns = ['case_id', 'start_time', 'end_time']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    # Get case attributes
    case_attributes = df.groupby('case_id').agg({
        'category': 'first',
        'priority': 'first',
        'severity': 'first',
        'activity': 'count'  # Number of activities
    }).reset_index()
    case_attributes.columns = ['case_id', 'category', 'priority', 'severity', 'num_activities']
    
    # Merge durations with attributes
    features_df = case_durations.merge(case_attributes, on='case_id')
    
    # Add temporal features
    features_df['start_hour'] = features_df['start_time'].dt.hour
    features_df['start_day_of_week'] = features_df['start_time'].dt.dayofweek
    features_df['start_month'] = features_df['start_time'].dt.month
    
    # Check if bug was reopened
    reopened_bugs = df[df['activity'].str.lower().str.contains('reopen', na=False)]['case_id'].unique()
    features_df['was_reopened'] = features_df['case_id'].isin(reopened_bugs).astype(int)
    
    # Historical statistics by category (at time of prediction)
    category_stats = calculate_historical_stats(df, 'category')
    features_df = features_df.merge(category_stats, on='category', how='left', suffixes=('', '_cat'))
    
    # Historical statistics by priority
    priority_stats = calculate_historical_stats(df, 'priority')
    features_df = features_df.merge(priority_stats, on='priority', how='left', suffixes=('', '_pri'))
    
    # Fill NaN values
    features_df = features_df.fillna(0)
    
    return features_df


def calculate_historical_stats(df, group_by_col):
    """
    Calculate historical statistics for a grouping column.
    
    Args:
        df: Event log dataframe
        group_by_col: Column to group by (e.g., 'category', 'priority')
        
    Returns:
        pd.DataFrame: Historical statistics
    """
    # Calculate case durations
    case_durations = df.groupby('case_id').agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    case_durations.columns = ['case_id', 'start_time', 'end_time']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    # Get group attribute
    case_group = df.groupby('case_id')[group_by_col].first().reset_index()
    case_durations = case_durations.merge(case_group, on='case_id')
    
    # Calculate statistics
    stats = case_durations.groupby(group_by_col).agg({
        'duration_hours': ['mean', 'median', 'std', 'count']
    }).reset_index()
    
    stats.columns = [
        group_by_col,
        f'avg_duration_{group_by_col}',
        f'median_duration_{group_by_col}',
        f'std_duration_{group_by_col}',
        f'count_{group_by_col}'
    ]
    
    return stats


def prepare_features_for_prediction(
    category, 
    priority, 
    severity, 
    historical_data=None,
    start_hour=9,
    start_day_of_week=0,
    num_similar_bugs=0
):
    """
    Prepare features for a single prediction.
    
    Args:
        category: Bug category
        priority: Bug priority
        severity: Bug severity
        historical_data: Historical DataFrame for calculating stats
        start_hour: Hour when bug was reported (0-23)
        start_day_of_week: Day of week (0=Monday, 6=Sunday)
        num_similar_bugs: Number of similar bugs in the past
        
    Returns:
        pd.DataFrame: Single row with features
    """
    features = {
        'category': category,
        'priority': priority,
        'severity': severity,
        'start_hour': start_hour,
        'start_day_of_week': start_day_of_week,
        'start_month': datetime.now().month,
        'num_activities': 5,  # Average number of activities
        'was_reopened': 0,  # Assume not reopened for new bug
        'num_similar_bugs': num_similar_bugs
    }
    
    # Add historical statistics if available
    if historical_data is not None and not historical_data.empty:
        # Category stats
        cat_stats = historical_data[historical_data['category'] == category]
        if not cat_stats.empty:
            features['avg_duration_category'] = cat_stats['duration_hours'].mean()
            features['median_duration_category'] = cat_stats['duration_hours'].median()
            features['std_duration_category'] = cat_stats['duration_hours'].std()
            features['count_category'] = len(cat_stats)
        else:
            features['avg_duration_category'] = 0
            features['median_duration_category'] = 0
            features['std_duration_category'] = 0
            features['count_category'] = 0
        
        # Priority stats
        pri_stats = historical_data[historical_data['priority'] == priority]
        if not pri_stats.empty:
            features['avg_duration_priority'] = pri_stats['duration_hours'].mean()
            features['median_duration_priority'] = pri_stats['duration_hours'].median()
            features['std_duration_priority'] = pri_stats['duration_hours'].std()
            features['count_priority'] = len(pri_stats)
        else:
            features['avg_duration_priority'] = 0
            features['median_duration_priority'] = 0
            features['std_duration_priority'] = 0
            features['count_priority'] = 0
    else:
        # Default values if no historical data
        features['avg_duration_category'] = 0
        features['median_duration_category'] = 0
        features['std_duration_category'] = 0
        features['count_category'] = 0
        features['avg_duration_priority'] = 0
        features['median_duration_priority'] = 0
        features['std_duration_priority'] = 0
        features['count_priority'] = 0
    
    return pd.DataFrame([features])


def encode_categorical_features(df, fit=True, encoders=None):
    """
    Encode categorical features for ML models.
    
    Args:
        df: Features dataframe
        fit: Whether to fit encoders or use existing ones
        encoders: Pre-fitted encoders (if fit=False)
        
    Returns:
        tuple: (encoded_df, encoders_dict)
    """
    from sklearn.preprocessing import LabelEncoder
    
    df_encoded = df.copy()
    
    if encoders is None:
        encoders = {}
    
    categorical_cols = ['category', 'priority', 'severity']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            if fit:
                encoders[col] = LabelEncoder()
                # Handle unseen categories
                df_encoded[col] = df_encoded[col].fillna('Unknown')
                df_encoded[f'{col}_encoded'] = encoders[col].fit_transform(df_encoded[col])
            else:
                if col in encoders:
                    df_encoded[col] = df_encoded[col].fillna('Unknown')
                    # Handle unseen categories during prediction
                    try:
                        df_encoded[f'{col}_encoded'] = encoders[col].transform(df_encoded[col])
                    except ValueError:
                        # If category is unseen, assign a default value (most frequent)
                        df_encoded[f'{col}_encoded'] = 0
                else:
                    df_encoded[f'{col}_encoded'] = 0
    
    return df_encoded, encoders


def get_feature_columns():
    """
    Get list of feature columns for ML models.
    
    Returns:
        list: Feature column names
    """
    return [
        'category_encoded',
        'priority_encoded',
        'severity_encoded',
        'num_activities',
        'start_hour',
        'start_day_of_week',
        'start_month',
        'was_reopened',
        'avg_duration_category',
        'median_duration_category',
        'std_duration_category',
        'count_category',
        'avg_duration_priority',
        'median_duration_priority',
        'std_duration_priority',
        'count_priority'
    ]


def calculate_complexity_score(predicted_duration, historical_avg, historical_std):
    """
    Calculate complexity/risk score based on prediction vs historical data.
    
    Args:
        predicted_duration: Predicted duration in hours
        historical_avg: Historical average duration
        historical_std: Historical standard deviation
        
    Returns:
        tuple: (complexity_score, risk_level)
            complexity_score: 0-100 score
            risk_level: 'Low', 'Medium', 'High', 'Critical'
    """
    if historical_std == 0 or pd.isna(historical_std):
        historical_std = historical_avg * 0.3  # Assume 30% std if no data
    
    # Calculate z-score
    if historical_avg > 0:
        z_score = (predicted_duration - historical_avg) / historical_std
    else:
        z_score = 0
    
    # Convert to 0-100 scale
    complexity_score = min(100, max(0, 50 + (z_score * 20)))
    
    # Determine risk level
    if complexity_score < 30:
        risk_level = 'Low'
    elif complexity_score < 60:
        risk_level = 'Medium'
    elif complexity_score < 80:
        risk_level = 'High'
    else:
        risk_level = 'Critical'
    
    return complexity_score, risk_level


def calculate_process_deviation(df, case_id=None, category=None, priority=None):
    """
    Calculate process deviation indicator (complexity indicator).
    Measures how much a case deviates from the standard process.
    
    Args:
        df: Event log dataframe
        case_id: Specific case to analyze (optional)
        category: Category to analyze (optional)
        priority: Priority to analyze (optional)
        
    Returns:
        dict: Deviation metrics
    """
    if df.empty:
        return {'deviation_score': 0, 'deviation_level': 'Normal', 'factors': []}
    
    # Get cases to analyze
    if case_id:
        cases_to_analyze = df[df['case_id'] == case_id]
    elif category:
        cases_to_analyze = df[df['category'] == category].copy()
    elif priority:
        cases_to_analyze = df[df['priority'] == priority].copy()
    else:
        cases_to_analyze = df.copy()
    
    if cases_to_analyze.empty:
        return {'deviation_score': 0, 'deviation_level': 'Normal', 'factors': []}
    
    # Calculate standard process metrics
    all_case_stats = df.groupby('case_id').agg({
        'activity': 'count',
        'timestamp': ['min', 'max']
    }).reset_index()
    all_case_stats.columns = ['case_id', 'num_activities', 'start_time', 'end_time']
    all_case_stats['duration_hours'] = (
        all_case_stats['end_time'] - all_case_stats['start_time']
    ).dt.total_seconds() / 3600
    
    standard_avg_activities = all_case_stats['num_activities'].mean()
    standard_avg_duration = all_case_stats['duration_hours'].mean()
    
    # Analyze target cases
    target_stats = cases_to_analyze.groupby('case_id').agg({
        'activity': 'count',
        'timestamp': ['min', 'max']
    }).reset_index()
    target_stats.columns = ['case_id', 'num_activities', 'start_time', 'end_time']
    target_stats['duration_hours'] = (
        target_stats['end_time'] - target_stats['start_time']
    ).dt.total_seconds() / 3600
    
    avg_activities = target_stats['num_activities'].mean()
    avg_duration = target_stats['duration_hours'].mean()
    
    # Check for rework (reopened bugs)
    reopened_cases = cases_to_analyze[
        cases_to_analyze['activity'].str.lower().str.contains('reopen', na=False)
    ]['case_id'].nunique()
    total_cases = cases_to_analyze['case_id'].nunique()
    rework_rate = (reopened_cases / total_cases * 100) if total_cases > 0 else 0
    
    # Calculate deviation factors
    factors = []
    deviation_score = 0
    
    # Activity count deviation
    if standard_avg_activities > 0:
        activity_deviation = abs(avg_activities - standard_avg_activities) / standard_avg_activities
        if activity_deviation > 0.2:  # More than 20% deviation
            factors.append(f"Activity count deviation: {activity_deviation*100:.1f}%")
            deviation_score += min(30, activity_deviation * 100)
    
    # Duration deviation
    if standard_avg_duration > 0:
        duration_deviation = abs(avg_duration - standard_avg_duration) / standard_avg_duration
        if duration_deviation > 0.2:  # More than 20% deviation
            factors.append(f"Duration deviation: {duration_deviation*100:.1f}%")
            deviation_score += min(40, duration_deviation * 100)
    
    # Rework rate
    if rework_rate > 10:  # More than 10% rework
        factors.append(f"High rework rate: {rework_rate:.1f}%")
        deviation_score += min(30, rework_rate)
    
    # Determine deviation level
    if deviation_score < 20:
        deviation_level = 'Normal'
    elif deviation_score < 50:
        deviation_level = 'Moderate'
    elif deviation_score < 80:
        deviation_level = 'High'
    else:
        deviation_level = 'Critical'
    
    return {
        'deviation_score': min(100, deviation_score),
        'deviation_level': deviation_level,
        'factors': factors,
        'avg_activities': avg_activities,
        'standard_avg_activities': standard_avg_activities,
        'rework_rate': rework_rate
    }

