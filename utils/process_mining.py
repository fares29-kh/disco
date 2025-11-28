"""
Process Mining Module using pm4py
"""

import pandas as pd
import numpy as np
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.filtering.log.variants import variants_filter
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.conversion.log import converter as log_converter


def prepare_event_log(df):
    """
    Prepare event log for pm4py processing.
    
    Args:
        df: Event log dataframe
        
    Returns:
        Event log object compatible with pm4py
    """
    # Rename columns to pm4py standard
    log_df = df.copy()
    log_df = log_df.rename(columns={
        'case_id': 'case:concept:name',
        'activity': 'concept:name',
        'timestamp': 'time:timestamp'
    })
    
    # Add required attributes
    log_df = dataframe_utils.convert_timestamp_columns_in_df(log_df)
    log_df = log_df.sort_values('time:timestamp')
    
    # Convert to event log
    event_log = log_converter.apply(log_df)
    
    return event_log


def compute_dfg(df):
    """
    Compute Directly-Follows Graph (DFG) with frequency and performance metrics.
    
    Args:
        df: Event log dataframe
        
    Returns:
        tuple: (dfg_frequency, dfg_performance, start_activities, end_activities)
    """
    event_log = prepare_event_log(df)
    
    # Calculate DFG frequency (number of times each edge is traversed)
    dfg_frequency = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.FREQUENCY)
    
    # Calculate DFG performance (average time between activities)
    dfg_performance = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.PERFORMANCE)
    
    # Get start and end activities
    from pm4py.statistics.start_activities.log import get as start_activities_get
    from pm4py.statistics.end_activities.log import get as end_activities_get
    
    start_activities = start_activities_get.get_start_activities(event_log)
    end_activities = end_activities_get.get_end_activities(event_log)
    
    return dfg_frequency, dfg_performance, start_activities, end_activities


def compute_dfg_with_colors(df, sla_threshold):
    """
    Compute DFG with color coding based on SLA threshold.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        
    Returns:
        dict: Dictionary containing DFG data with colors
    """
    dfg_freq, dfg_perf, start_act, end_act = compute_dfg(df)
    
    # Prepare edges with colors
    edges = []
    for edge, frequency in dfg_freq.items():
        from_activity, to_activity = edge
        
        # Get performance (in seconds)
        performance_seconds = dfg_perf.get(edge, 0)
        performance_hours = performance_seconds / 3600
        
        # Determine color based on SLA
        if performance_hours > sla_threshold:
            color = 'red'
            status = 'critical'
        elif performance_hours > sla_threshold * 0.5:
            color = 'orange'
            status = 'warning'
        else:
            color = 'green'
            status = 'ok'
        
        edges.append({
            'from': from_activity,
            'to': to_activity,
            'frequency': frequency,
            'performance_hours': performance_hours,
            'performance_seconds': performance_seconds,
            'color': color,
            'status': status
        })
    
    # Calculate average duration for each activity (node)
    from utils.metrics import calculate_activity_durations
    activity_durations_df = calculate_activity_durations(df)
    
    # Calculate average duration per activity
    node_durations = {}
    if not activity_durations_df.empty:
        avg_durations = activity_durations_df.groupby('activity')['activity_duration_hours'].mean()
        for activity, duration in avg_durations.items():
            # Determine color based on 24h threshold
            if duration > 24:
                node_color = 'red'
                node_status = 'critical'
            else:
                node_color = 'blue'
                node_status = 'ok'
            
            node_durations[activity] = {
                'duration_hours': duration,
                'color': node_color,
                'status': node_status
            }
    
    # Add default values for activities without duration data
    all_activities = set([edge['from'] for edge in edges] + [edge['to'] for edge in edges])
    for activity in all_activities:
        if activity not in node_durations:
            node_durations[activity] = {
                'duration_hours': 0,
                'color': 'blue',
                'status': 'unknown'
            }
    
    # Force specific activities as start/end nodes
    # Override start and end activities to only show specific nodes
    forced_start_activities = {}
    forced_end_activities = {}
    
    for activity in all_activities:
        activity_lower = activity.lower().strip()
        
        # Only "Report the bug" should be green (start)
        # Must start with "report" to avoid matching "validate bug report"
        if activity_lower.startswith('report') and 'bug' in activity_lower:
            forced_start_activities[activity] = start_act.get(activity, 1)
        
        # Only "Close Bug" should be violet (end)
        # Must start with "close" to be precise
        if activity_lower.startswith('close') and 'bug' in activity_lower:
            forced_end_activities[activity] = end_act.get(activity, 1)
    
    # Replace the original start/end activities with forced ones
    start_act = forced_start_activities
    end_act = forced_end_activities
    
    return {
        'edges': edges,
        'start_activities': start_act,
        'end_activities': end_act,
        'nodes': node_durations
    }


def get_process_variants(df, top_n=10):
    """
    Get the most frequent process variants.
    
    Args:
        df: Event log dataframe
        top_n: Number of top variants to return
        
    Returns:
        pd.DataFrame: DataFrame with variant information
    """
    event_log = prepare_event_log(df)
    
    # Get variants
    from pm4py.statistics.variants.log import get as variants_get
    variants = variants_get.get_variants(event_log)
    
    # Convert to DataFrame
    variant_list = []
    for variant_key, cases in variants.items():
        # Convert variant to string - it can be a tuple or string
        if isinstance(variant_key, tuple):
            variant_str = ' → '.join(str(activity) for activity in variant_key)
        else:
            variant_str = str(variant_key).replace(',', ' → ')
        
        variant_list.append({
            'variant': variant_str,
            'frequency': len(cases),
            'percentage': (len(cases) / len(event_log)) * 100
        })
    
    if variant_list:
        variant_df = pd.DataFrame(variant_list)
        variant_df = variant_df.sort_values('frequency', ascending=False).head(top_n)
    else:
        # Return empty DataFrame with correct columns if no variants
        variant_df = pd.DataFrame(columns=['variant', 'frequency', 'percentage'])
    
    return variant_df


def calculate_process_statistics(df):
    """
    Calculate various process statistics.
    
    Args:
        df: Event log dataframe
        
    Returns:
        dict: Dictionary containing process statistics
    """
    event_log = prepare_event_log(df)
    
    stats = {}
    
    # Number of cases
    stats['num_cases'] = len(set([trace.attributes['concept:name'] for trace in event_log]))
    
    # Number of events
    stats['num_events'] = sum([len(trace) for trace in event_log])
    
    # Number of activities
    stats['num_activities'] = len(set([event['concept:name'] for trace in event_log for event in trace]))
    
    # Average events per case
    stats['avg_events_per_case'] = stats['num_events'] / stats['num_cases'] if stats['num_cases'] > 0 else 0
    
    return stats


def compute_activity_positions(edges):
    """
    Compute positions for activities in the process graph for visualization.
    
    Args:
        edges: List of edge dictionaries
        
    Returns:
        dict: Dictionary mapping activity names to (x, y) positions
    """
    # Get all unique activities
    activities = set()
    for edge in edges:
        activities.add(edge['from'])
        activities.add(edge['to'])
    
    activities = sorted(list(activities))
    
    # Simple left-to-right layout
    positions = {}
    num_activities = len(activities)
    
    # Try to detect common workflow patterns
    start_activities = ['Open', 'Created', 'New', 'Start']
    end_activities = ['Close', 'Closed', 'Done', 'Resolved', 'Complete']
    
    # Categorize activities
    start = []
    middle = []
    end = []
    
    for activity in activities:
        if any(s.lower() in activity.lower() for s in start_activities):
            start.append(activity)
        elif any(e.lower() in activity.lower() for e in end_activities):
            end.append(activity)
        else:
            middle.append(activity)
    
    # Assign positions
    all_ordered = start + middle + end
    x_spacing = 1.0 / (len(all_ordered) + 1)
    
    for i, activity in enumerate(all_ordered):
        positions[activity] = {
            'x': (i + 1) * x_spacing,
            'y': 0.5 + (np.random.random() - 0.5) * 0.3  # Add some vertical variation
        }
    
    return positions


def analyze_loops(df):
    """
    Detect loops/rework in the process (activities that repeat within a case).
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Information about loops found
    """
    loops = []
    
    for case_id, group in df.groupby('case_id'):
        activities = group['activity'].tolist()
        activity_counts = pd.Series(activities).value_counts()
        
        # Find activities that occur more than once
        repeated_activities = activity_counts[activity_counts > 1]
        
        if len(repeated_activities) > 0:
            for activity, count in repeated_activities.items():
                loops.append({
                    'case_id': case_id,
                    'activity': activity,
                    'repetitions': count
                })
    
    if loops:
        loops_df = pd.DataFrame(loops)
        
        # Aggregate statistics
        loop_summary = loops_df.groupby('activity').agg({
            'repetitions': ['sum', 'mean', 'count']
        }).reset_index()
        
        loop_summary.columns = ['activity', 'total_repetitions', 'avg_repetitions', 'cases_affected']
        loop_summary = loop_summary.sort_values('total_repetitions', ascending=False)
        
        return loop_summary
    else:
        return pd.DataFrame(columns=['activity', 'total_repetitions', 'avg_repetitions', 'cases_affected'])


def calculate_resource_utilization(df):
    """
    Calculate resource utilization if assignee/team information is available.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Resource utilization statistics
    """
    if 'assignee' not in df.columns:
        return None
    
    # Calculate workload per assignee
    workload = df.groupby('assignee').agg({
        'case_id': 'nunique',
        'activity': 'count'
    }).reset_index()
    
    workload.columns = ['assignee', 'cases_handled', 'activities_performed']
    workload = workload.sort_values('cases_handled', ascending=False)
    
    return workload


def detect_parallel_activities(df):
    """
    Detect activities that often happen in parallel.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Information about parallel activities
    """
    # Group by case and find activities with overlapping timestamps
    parallel_activities = []
    
    for case_id, group in df.groupby('case_id'):
        activities = group.sort_values('timestamp')
        
        for i in range(len(activities) - 1):
            current_activity = activities.iloc[i]
            next_activity = activities.iloc[i + 1]
            
            # If activities are very close in time (within 1 hour), consider them parallel
            time_diff = (next_activity['timestamp'] - current_activity['timestamp']).total_seconds() / 3600
            
            if time_diff < 1:
                parallel_activities.append({
                    'activity_1': current_activity['activity'],
                    'activity_2': next_activity['activity'],
                    'time_diff_hours': time_diff
                })
    
    if parallel_activities:
        parallel_df = pd.DataFrame(parallel_activities)
        
        # Count occurrences
        parallel_summary = parallel_df.groupby(['activity_1', 'activity_2']).size().reset_index(name='frequency')
        parallel_summary = parallel_summary.sort_values('frequency', ascending=False)
        
        return parallel_summary
    else:
        return pd.DataFrame(columns=['activity_1', 'activity_2', 'frequency'])

