"""
AI-Based Bug Category Prioritization Module
"""

import pandas as pd
import numpy as np
from utils.feature_engineering import extract_features_from_log, calculate_process_deviation
from utils.ml_models import BugDurationPredictor


def prioritize_categories(df, sla_threshold=24.0, predictor=None):
    """
    Prioritize bug categories based on AI analysis.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        predictor: Trained BugDurationPredictor (optional)
        
    Returns:
        pd.DataFrame: Category prioritization ranking with scores
    """
    if df.empty or 'category' not in df.columns:
        return pd.DataFrame()
    
    # Get unique categories
    categories = df['category'].dropna().unique()
    
    if len(categories) == 0:
        return pd.DataFrame()
    
    # Calculate statistics per category
    case_durations = df.groupby('case_id').agg({
        'timestamp': ['min', 'max'],
        'category': 'first'
    }).reset_index()
    case_durations.columns = ['case_id', 'start_time', 'end_time', 'category']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    results = []
    
    for category in categories:
        cat_data = df[df['category'] == category]
        cat_cases = case_durations[case_durations['category'] == category]
        
        if cat_cases.empty:
            continue
        
        # Basic statistics
        instance_count = cat_cases['case_id'].nunique()
        avg_duration = cat_cases['duration_hours'].mean()
        median_duration = cat_cases['duration_hours'].median()
        std_duration = cat_cases['duration_hours'].std()
        
        # Priority and severity distribution
        priority_dist = {}
        severity_dist = {}
        if 'priority' in df.columns:
            priority_dist = cat_data.groupby('case_id')['priority'].first().value_counts().to_dict()
        if 'severity' in df.columns:
            severity_dist = cat_data.groupby('case_id')['severity'].first().value_counts().to_dict()
        
        # Calculate predicted resolution time if predictor available
        predicted_resolution_time = avg_duration
        if predictor is not None and predictor.is_trained:
            try:
                from utils.feature_engineering import prepare_features_for_prediction, extract_features_from_log
                hist_data = extract_features_from_log(df)
                
                # Use most common priority/severity for this category
                most_common_priority = cat_data.groupby('case_id')['priority'].first().mode()[0] if 'priority' in df.columns else None
                most_common_severity = cat_data.groupby('case_id')['severity'].first().mode()[0] if 'severity' in df.columns else None
                
                if most_common_priority and most_common_severity:
                    features = prepare_features_for_prediction(
                        category=category,
                        priority=most_common_priority,
                        severity=most_common_severity,
                        historical_data=hist_data
                    )
                    predicted_resolution_time = predictor.predict(features)[0]
            except:
                predicted_resolution_time = avg_duration
        
        # Calculate delay risk (probability of exceeding SLA)
        delay_risk = 0
        if sla_threshold > 0:
            cases_exceeding_sla = len(cat_cases[cat_cases['duration_hours'] > sla_threshold])
            delay_risk = (cases_exceeding_sla / len(cat_cases)) * 100 if len(cat_cases) > 0 else 0
        
        # Calculate process deviation
        deviation = calculate_process_deviation(df, category=category)
        deviation_score = deviation.get('deviation_score', 0)
        
        # Calculate priority score (0-100)
        # Factors: delay risk (40%), duration (30%), deviation (20%), instance count (10%)
        priority_score = (
            min(40, delay_risk) * 0.4 +
            min(30, (avg_duration / sla_threshold * 30) if sla_threshold > 0 else 0) +
            min(20, deviation_score * 0.2) +
            min(10, (instance_count / max(df['case_id'].nunique(), 1) * 100) * 0.1)
        )
        
        # Determine suggested action
        if priority_score >= 70:
            suggested_action = "Handle First"
        elif priority_score >= 40:
            suggested_action = "Schedule Normally"
        else:
            suggested_action = "Can Defer"
        
        results.append({
            'category': category,
            'priority_score': priority_score,
            'predicted_resolution_time': predicted_resolution_time,
            'predicted_delay_risk': delay_risk,
            'avg_duration': avg_duration,
            'instance_count': instance_count,
            'deviation_score': deviation_score,
            'suggested_action': suggested_action,
            'median_duration': median_duration,
            'std_duration': std_duration
        })
    
    # Create DataFrame and sort by priority score
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('priority_score', ascending=False)
        result_df = result_df.reset_index(drop=True)
    
    return result_df


def analyze_overall_process_performance(df, sla_threshold=24.0):
    """
    Analyze overall process performance by category.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        
    Returns:
        dict: Overall performance metrics and category impacts
    """
    if df.empty:
        return {}
    
    # Calculate overall KPIs
    case_durations = df.groupby('case_id').agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    case_durations.columns = ['case_id', 'start_time', 'end_time']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    overall_avg_duration = case_durations['duration_hours'].mean()
    overall_median_duration = case_durations['duration_hours'].median()
    cases_exceeding_sla = len(case_durations[case_durations['duration_hours'] > sla_threshold])
    sla_breach_rate = (cases_exceeding_sla / len(case_durations)) * 100 if len(case_durations) > 0 else 0
    
    # Count reassignments (activity changes that suggest reassignment)
    reassignments = df.groupby('case_id')['activity'].nunique()
    avg_reassignments = reassignments.mean()
    max_reassignments = reassignments.max()
    
    # Check for loops (rework)
    reopened_cases = df[df['activity'].str.lower().str.contains('reopen', na=False)]['case_id'].nunique()
    rework_rate = (reopened_cases / df['case_id'].nunique()) * 100 if df['case_id'].nunique() > 0 else 0
    
    # Category impact analysis
    category_impacts = []
    if 'category' in df.columns:
        categories = df['category'].dropna().unique()
        case_durations_with_cat = case_durations.merge(
            df.groupby('case_id')['category'].first().reset_index(),
            on='case_id',
            how='left'
        )
        
        for category in categories:
            cat_cases = case_durations_with_cat[case_durations_with_cat['category'] == category]
            if cat_cases.empty:
                continue
            
            cat_avg = cat_cases['duration_hours'].mean()
            cat_count = len(cat_cases)
            cat_impact = cat_count * cat_avg  # Total hours consumed
            
            category_impacts.append({
                'category': category,
                'instance_count': cat_count,
                'avg_duration': cat_avg,
                'total_impact_hours': cat_impact,
                'impact_percentage': (cat_impact / (overall_avg_duration * len(case_durations))) * 100 if len(case_durations) > 0 else 0,
                'sla_breach_count': len(cat_cases[cat_cases['duration_hours'] > sla_threshold]),
                'sla_breach_rate': (len(cat_cases[cat_cases['duration_hours'] > sla_threshold]) / len(cat_cases)) * 100 if len(cat_cases) > 0 else 0
            })
        
        category_impacts_df = pd.DataFrame(category_impacts)
        if not category_impacts_df.empty:
            category_impacts_df = category_impacts_df.sort_values('total_impact_hours', ascending=False)
    
    # Most critical activities (slowest activities)
    activity_durations = df.groupby(['case_id', 'activity']).agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    activity_durations.columns = ['case_id', 'activity', 'start_time', 'end_time']
    activity_durations['duration_hours'] = (
        activity_durations['end_time'] - activity_durations['start_time']
    ).dt.total_seconds() / 3600
    
    critical_activities = activity_durations.groupby('activity').agg({
        'duration_hours': ['mean', 'count']
    }).reset_index()
    critical_activities.columns = ['activity', 'avg_duration', 'frequency']
    critical_activities = critical_activities.sort_values('avg_duration', ascending=False).head(10)
    
    return {
        'overall_avg_duration': overall_avg_duration,
        'overall_median_duration': overall_median_duration,
        'sla_breach_rate': sla_breach_rate,
        'sla_breach_count': cases_exceeding_sla,
        'avg_reassignments': avg_reassignments,
        'max_reassignments': max_reassignments,
        'rework_rate': rework_rate,
        'total_cases': len(case_durations),
        'category_impacts': category_impacts_df if 'category_impacts_df' in locals() else pd.DataFrame(),
        'critical_activities': critical_activities
    }

