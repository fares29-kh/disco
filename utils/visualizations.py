"""
Visualization Module using Plotly and Graphviz
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta


def plot_process_map(dfg_data, width=1200, height=600, animated=False):
    """
    Create an interactive process map using Plotly with duration labels.
    
    Args:
        dfg_data: Dictionary containing edges, start_activities, end_activities
        width: Figure width
        height: Figure height
        
    Returns:
        plotly.graph_objects.Figure
    """
    edges = dfg_data['edges']
    
    if not edges:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No process data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges with attributes
    for edge in edges:
        G.add_edge(
            edge['from'], 
            edge['to'],
            frequency=edge['frequency'],
            performance=edge['performance_hours'],
            color=edge['color']
        )
    
    # Calculate layout
    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create edge traces and annotations for durations
    edge_traces = []
    edge_annotations = []
    
    for edge in edges:
        x0, y0 = pos[edge['from']]
        x1, y1 = pos[edge['to']]
        
        # Determine color based on status
        color_map = {
            'green': '#2ECC71',    # Vert plus clair
            'orange': '#F39C12',   # Orange
            'red': '#E74C3C'       # Rouge
        }
        edge_color = color_map.get(edge['color'], '#95A5A6')
        
        # Calculate edge width based on frequency (normalized)
        max_freq = max([e['frequency'] for e in edges])
        edge_width = max(2, min(edge['frequency'] / max_freq * 15, 15))
        
        # Edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                width=edge_width,
                color=edge_color
            ),
            hoverinfo='text',
            text=f"<b>{edge['from']} → {edge['to']}</b><br>"
                 f"Frequency: <b>{edge['frequency']}</b> times<br>"
                 f"Avg Duration: <b>{edge['performance_hours']:.1f}h</b><br>"
                 f"Status: <b>{edge['status'].upper()}</b>",
            showlegend=False,
            opacity=0.7
        )
        edge_traces.append(edge_trace)
        
        # Calculate midpoint for duration label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        
        # Format duration text
        duration_hours = edge['performance_hours']
        if duration_hours < 1:
            duration_text = f"{duration_hours * 60:.0f}min"
        elif duration_hours < 24:
            duration_text = f"{duration_hours:.1f}h"
        else:
            days = duration_hours / 24
            duration_text = f"{days:.1f}d"
        
        # Add annotation for duration
        edge_annotations.append(
            dict(
                x=mid_x,
                y=mid_y,
                text=f"<b>{duration_text}</b><br><span style='font-size:10px'>({edge['frequency']})</span>",
                showarrow=False,
                font=dict(
                    size=11,
                    color='black',
                    family="Arial Black"
                ),
                bgcolor='white',
                bordercolor=edge_color,
                borderwidth=2,
                borderpad=4,
                opacity=0.95
            )
        )
    
    # Get node duration data
    nodes_data = dfg_data.get('nodes', {})
    
    # Create figure with edges only first
    fig = go.Figure(data=edge_traces)
    
    # Get start and end activities
    start_activities = dfg_data.get('start_activities', {})
    end_activities = dfg_data.get('end_activities', {})
    
    # Add rectangular shapes for nodes using annotations
    for node in G.nodes():
        x, y = pos[node]
        node_info = nodes_data.get(node, {'duration_hours': 0, 'color': 'blue', 'status': 'unknown'})
        duration_hours = node_info['duration_hours']
        node_color_key = node_info['color']
        
        # Format duration text
        if duration_hours < 1:
            duration_text = f"{duration_hours * 60:.0f}min"
        elif duration_hours < 24:
            duration_text = f"{duration_hours:.1f}h"
        else:
            days = duration_hours / 24
            duration_text = f"{days:.1f}d"
        
        # Calculate node statistics for hover text
        in_freq = sum([e['frequency'] for e in edges if e['to'] == node])
        out_freq = sum([e['frequency'] for e in edges if e['from'] == node])
        total_freq = in_freq + out_freq
        
        # Check if this is a start or end node
        is_start = node in start_activities
        is_end = node in end_activities
        
        # Node color: Special colors for start/end, otherwise RED if > 24h, BLUE otherwise
        if is_start:
            color_hex = '#27AE60'  # Green for start
            border_color = '#1E8449'
            node_label = f"<b>▶️ START<br>{node}</b>"
            font_size = 12
            borderwidth = 5
            borderpad = 12
        elif is_end:
            color_hex = '#8E44AD'  # Purple for end
            border_color = '#6C3483'
            node_label = f"<b>{node}<br>⏹️ END</b>"
            font_size = 12
            borderwidth = 5
            borderpad = 12
        elif node_color_key == 'red':
            color_hex = '#E74C3C'  # Rouge
            border_color = '#C0392B'
            node_label = f"<b>{node}</b>"
            font_size = 11
            borderwidth = 3
            borderpad = 8
        else:
            color_hex = '#3498DB'  # Bleu
            border_color = '#2980B9'
            node_label = f"<b>{node}</b>"
            font_size = 11
            borderwidth = 3
            borderpad = 8
        
        # Create rectangular annotation for the node
        fig.add_annotation(
            x=x,
            y=y,
            text=node_label,
            showarrow=False,
            font=dict(
                size=font_size,
                color='white',
                family='Arial Black'
            ),
            bgcolor=color_hex,
            bordercolor=border_color,
            borderwidth=borderwidth,
            borderpad=borderpad,
            opacity=0.98 if (is_start or is_end) else 0.95,
            hovertext=(
                f"<b>{node}</b><br>"
                + (f"🚀 <b>START NODE</b> (Entry Point)<br>" if is_start else "")
                + (f"🏁 <b>END NODE</b> (Exit Point)<br>" if is_end else "")
                + f"Avg Duration: <b>{duration_text}</b><br>"
                + f"Status: {'⚠️ CRITICAL' if node_color_key == 'red' else '✅ Normal'}<br>"
                + f"In: {in_freq} | Out: {out_freq}<br>"
                + f"Total: {total_freq}"
            ),
            hoverlabel=dict(
                bgcolor='white',
                font_size=12,
                font_color='black'
            )
        )
        
        # Add special badge/label above start and end nodes
        if is_start:
            fig.add_annotation(
                x=x,
                y=y + 0.15,  # Above the node
                text="<b>🚀 START</b>",
                showarrow=False,
                font=dict(
                    size=10,
                    color='white',
                    family='Arial Black'
                ),
                bgcolor='#1E8449',
                bordercolor='#27AE60',
                borderwidth=2,
                borderpad=5,
                opacity=1.0
            )
        elif is_end:
            fig.add_annotation(
                x=x,
                y=y + 0.15,  # Above the node
                text="<b>🏁 END</b>",
                showarrow=False,
                font=dict(
                    size=10,
                    color='white',
                    family='Arial Black'
                ),
                bgcolor='#6C3483',
                bordercolor='#8E44AD',
                borderwidth=2,
                borderpad=5,
                opacity=1.0
            )
        
        # Add duration annotation below the rectangle
        if duration_hours > 0:
            fig.add_annotation(
                x=x,
                y=y - 0.12,  # Below the rectangle
                text=f"<b>{duration_text}</b>",
                showarrow=False,
                font=dict(
                    size=10,
                    color=color_hex,
                    family="Arial Black"
                ),
                bgcolor='white',
                bordercolor=color_hex,
                borderwidth=2,
                borderpad=3,
                opacity=0.95
            )
    
    # Add duration annotations for edges
    for annotation in edge_annotations:
        fig.add_annotation(annotation)
    
    # Add animated tokens if requested
    if animated:
        # Create animation frames with tokens moving along edges
        frames = []
        num_frames = 120  # Increased frames for smoother, slower animation
        
        # Define speed multipliers based on edge color (SLOWER overall)
        speed_map = {
            'green': 2.0,   # Fast (vert) - tokens move quickly (reduced from 3.0)
            'orange': 0.7,  # Normal (orange) - normal speed (reduced from 1.0)
            'red': 0.35     # Slow (rouge) - tokens move slowly (reduced from 0.5)
        }
        
        # Token density: how many tokens per edge to show simultaneously
        TOKENS_PER_EDGE = 8  # Number of tokens visible on each edge at once
        
        # Create token traces for each edge
        token_traces = []
        for edge_idx, edge in enumerate(edges):
            x0, y0 = pos[edge['from']]
            x1, y1 = pos[edge['to']]
            edge_color = color_map.get(edge['color'], '#95A5A6')
            speed = speed_map.get(edge['color'], 0.7)
            
            # Create multiple tokens per edge to show continuous flow
            # Each token will follow the previous one with a spacing
            num_tokens = TOKENS_PER_EDGE
            
            for token_idx in range(num_tokens):
                # NO perpendicular offset - tokens must be exactly on the line
                token_offset = 0.0
                
                # Calculate spacing between tokens (they should be evenly distributed)
                # Each token starts at a different position along the line
                token_spacing = 1.0 / num_tokens  # Spacing between tokens (0 to 1)
                base_phase = token_idx * token_spacing  # Starting position (0 to 1)
                
                token_trace = go.Scatter(
                    x=[],
                    y=[],
                    mode='markers',
                    marker=dict(
                        size=9,  # Slightly larger for better visibility
                        color=edge_color,
                        line=dict(width=1.5, color='white'),
                        opacity=0.9
                    ),
                    name=f"Token {edge_idx}-{token_idx}",
                    showlegend=False,
                    hoverinfo='skip'
                )
                token_traces.append({
                    'trace': token_trace,
                    'x0': x0,
                    'y0': y0,
                    'x1': x1,
                    'y1': y1,
                    'speed': speed,
                    'offset': token_offset,
                    'color': edge_color,
                    'edge_idx': edge_idx,
                    'base_phase': base_phase  # Starting position along the edge
                })
        
        # Create frames for animation
        for frame_num in range(num_frames):
            frame_data = list(edge_traces)  # Start with static edges
            
            # Calculate token positions for this frame
            for token_info in token_traces:
                # Calculate progress along edge (0 to 1)
                # Speed affects how fast tokens progress through frames
                # SLOWER: use more frames per cycle
                # The speed multiplier affects how quickly tokens move
                base_cycle_length = 100  # Base cycle length (frames for one complete journey)
                cycle_length = max(30, int(base_cycle_length / token_info['speed']))  # Slower tokens have longer cycles
                
                # Calculate the current position of this token
                # Tokens move continuously along the edge, wrapping around
                # base_phase gives the starting offset, then we add movement
                # Speed determines how fast each token progresses
                movement_speed = token_info['speed'] / base_cycle_length  # Speed-dependent movement per frame
                total_progress = (token_info['base_phase'] + frame_num * movement_speed) % 1.0
                
                # Calculate position along edge - EXACTLY on the line
                x_pos = token_info['x0'] + (token_info['x1'] - token_info['x0']) * total_progress
                y_pos = token_info['y0'] + (token_info['y1'] - token_info['y0']) * total_progress
                
                # NO perpendicular offset - tokens must stay exactly on the line
                # All tokens move on the same line path
                
                # Always show token (it wraps around)
                token_trace = go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode='markers',
                    marker=dict(
                        size=9,  # Slightly larger for better visibility
                        color=token_info['color'],
                        line=dict(width=1.5, color='white'),
                        opacity=0.9
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
                frame_data.append(token_trace)
            
            frames.append(go.Frame(data=frame_data, name=str(frame_num)))
        
        # Add initial token positions (distributed along edges)
        for token_info in token_traces:
            # Calculate initial position based on base_phase - EXACTLY on the line
            x_init = token_info['x0'] + (token_info['x1'] - token_info['x0']) * token_info['base_phase']
            y_init = token_info['y0'] + (token_info['y1'] - token_info['y0']) * token_info['base_phase']
            
            # NO perpendicular offset - tokens must be exactly on the line
            
            token_trace = go.Scatter(
                x=[x_init],
                y=[y_init],
                mode='markers',
                marker=dict(
                    size=9,  # Slightly larger for better visibility
                    color=token_info['color'],
                    line=dict(width=1.5, color='white'),
                    opacity=0.9
                ),
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_trace(token_trace)
        
        # Update layout with animation controls
        fig.update_layout(
            title={
                'text': "Process Map - Workflow with Duration (🟢=Start, 🟣=End)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
            width=width,
            height=height,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 80, 'redraw': True},  # Slower frame duration
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.0,
                'y': 0.0,
                'xanchor': 'left',
                'yanchor': 'bottom',
                'pad': {'t': 50, 'r': 10}
            }],
            sliders=[{
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 14},
                    'prefix': 'Frame:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 0},
                'pad': {'b': 10, 't': 50},
                'len': 0.9,
                'x': 0.1,
                'y': 0,
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': str(i),
                        'method': 'animate'
                    }
                    for i, f in enumerate(frames)
                ]
            }]
        )
        
        fig.frames = frames
    else:
        # No animation - standard layout
        fig.update_layout(
            title={
                'text': "Process Map - Workflow with Duration (🟢=Start, 🟣=End)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
            width=width,
            height=height
        )
    
    # Add legend box for start/end nodes
    # Get the axis ranges to position the legend
    all_x = [pos[node][0] for node in G.nodes()]
    all_y = [pos[node][1] for node in G.nodes()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Position legend in top-left corner
    legend_x = x_min - 0.3
    legend_y = y_max + 0.3
    
    # Add legend background
    fig.add_annotation(
        x=legend_x,
        y=legend_y,
        text=(
            "<b>Legend:</b><br>"
            "🟢 <b>START</b> = Entry Point<br>"
            "🟣 <b>END</b> = Exit Point"
        ),
        showarrow=False,
        font=dict(size=10, color='#2C3E50', family='Arial'),
        bgcolor='rgba(255, 255, 255, 0.95)',
        bordercolor='#2C3E50',
        borderwidth=2,
        borderpad=8,
        align='left',
        xanchor='left',
        yanchor='top'
    )
    
    return fig


def plot_heatmap(heatmap_data, title="Average Duration Heatmap (hours)"):
    """
    Create heatmap of average durations.
    
    Args:
        heatmap_data: Pivot table with categories as index and activities as columns
        title: Chart title
        
    Returns:
        plotly.graph_objects.Figure
    """
    if heatmap_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No heatmap data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='RdYlGn_r',  # Red (slow) to Green (fast)
        text=np.round(heatmap_data.values, 2),
        texttemplate='%{text:.1f}h',
        textfont={"size": 10},
        colorbar=dict(title="Hours")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Activity",
        yaxis_title="Category",
        height=400,
        margin=dict(l=150)
    )
    
    return fig


def plot_duration_distribution(case_durations, sla_threshold):
    """
    Create histogram of case durations.
    
    Args:
        case_durations: DataFrame with duration_hours column
        sla_threshold: SLA threshold in hours
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=case_durations['duration_hours'],
        nbinsx=30,
        name='Cases',
        marker_color='#4A90E2'
    ))
    
    # Add SLA threshold line
    fig.add_vline(
        x=sla_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"SLA: {sla_threshold}h",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Distribution of Case Durations",
        xaxis_title="Duration (hours)",
        yaxis_title="Number of Cases",
        height=400
    )
    
    return fig


def plot_timeline(df, sla_threshold, max_cases=50):
    """
    Create Gantt chart timeline of cases.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        max_cases: Maximum number of cases to display
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Calculate case durations
    case_summary = df.groupby('case_id').agg({
        'timestamp': ['min', 'max'],
        'priority': 'first',
        'category': 'first'
    }).reset_index()
    
    case_summary.columns = ['case_id', 'start', 'end', 'priority', 'category']
    case_summary['duration_hours'] = (case_summary['end'] - case_summary['start']).dt.total_seconds() / 3600
    
    # Filter to at-risk cases
    at_risk = case_summary[case_summary['duration_hours'] > sla_threshold]
    at_risk = at_risk.sort_values('duration_hours', ascending=False).head(max_cases)
    
    if at_risk.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No cases exceeding SLA threshold",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Create Gantt chart
    fig = px.timeline(
        at_risk,
        x_start='start',
        x_end='end',
        y='case_id',
        color='category',
        hover_data=['priority', 'duration_hours'],
        title=f"Timeline of Critical Cases (>{sla_threshold}h)"
    )
    
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(height=max(400, len(at_risk) * 20))
    
    return fig


def plot_category_comparison(category_stats):
    """
    Create bar chart comparing categories.
    
    Args:
        category_stats: DataFrame with category statistics
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Average Duration by Category', 'Case Count by Category')
    )
    
    # Duration chart
    fig.add_trace(
        go.Bar(
            x=category_stats['category'],
            y=category_stats['avg_duration'],
            name='Avg Duration',
            marker_color='#FF6B6B'
        ),
        row=1, col=1
    )
    
    # Count chart
    fig.add_trace(
        go.Bar(
            x=category_stats['category'],
            y=category_stats['case_count'],
            name='Case Count',
            marker_color='#4ECDC4'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Category", row=1, col=1)
    fig.update_xaxes(title_text="Category", row=1, col=2)
    fig.update_yaxes(title_text="Hours", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig


def plot_activity_frequency(df):
    """
    Create bar chart of activity frequencies.
    
    Args:
        df: Event log dataframe
        
    Returns:
        plotly.graph_objects.Figure
    """
    activity_counts = df['activity'].value_counts().reset_index()
    activity_counts.columns = ['Activity', 'Frequency']
    
    fig = px.bar(
        activity_counts,
        x='Activity',
        y='Frequency',
        title='Activity Frequency Distribution',
        color='Frequency',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(height=400)
    
    return fig


def plot_temporal_analysis(df):
    """
    Create time series analysis of cases over time.
    
    Args:
        df: Event log dataframe
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Count cases starting per day
    daily_cases = df.groupby([df['timestamp'].dt.date, 'case_id']).size().reset_index()
    daily_cases.columns = ['date', 'case_id', 'events']
    daily_counts = daily_cases.groupby('date').size().reset_index()
    daily_counts.columns = ['date', 'cases_started']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_counts['date'],
        y=daily_counts['cases_started'],
        mode='lines+markers',
        name='Cases per Day',
        line=dict(color='#4A90E2', width=2)
    ))
    
    fig.update_layout(
        title='Cases Started Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Cases',
        height=400
    )
    
    return fig


def plot_priority_distribution(df):
    """
    Create pie chart of priority distribution.
    
    Args:
        df: Event log dataframe
        
    Returns:
        plotly.graph_objects.Figure
    """
    priority_counts = df.groupby('case_id')['priority'].first().value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=priority_counts.index,
        values=priority_counts.values,
        hole=0.3
    )])
    
    fig.update_layout(
        title='Bug Priority Distribution',
        height=400
    )
    
    return fig


def plot_variant_analysis(variant_df):
    """
    Create bar chart of top process variants.
    
    Args:
        variant_df: DataFrame with variant information
        
    Returns:
        plotly.graph_objects.Figure
    """
    if variant_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No variant data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Truncate long variant names
    variant_df['variant_short'] = variant_df['variant'].apply(
        lambda x: x[:50] + '...' if len(x) > 50 else x
    )
    
    fig = go.Figure(data=[
        go.Bar(
            x=variant_df['frequency'],
            y=variant_df['variant_short'],
            orientation='h',
            text=variant_df['percentage'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
            marker_color='#9B59B6'
        )
    ])
    
    fig.update_layout(
        title='Top Process Variants',
        xaxis_title='Frequency',
        yaxis_title='Process Path',
        height=max(400, len(variant_df) * 40),
        margin=dict(l=200)
    )
    
    return fig


def create_kpi_card(title, value, delta=None, delta_color="normal"):
    """
    Create HTML for a KPI card.
    
    Args:
        title: KPI title
        value: KPI value
        delta: Optional delta value
        delta_color: Color for delta (normal, inverse, off)
        
    Returns:
        str: HTML string for the card
    """
    delta_html = ""
    if delta is not None:
        color = "green" if delta > 0 else "red"
        if delta_color == "inverse":
            color = "red" if delta > 0 else "green"
        delta_html = f'<div style="color: {color}; font-size: 14px;">{"↑" if delta > 0 else "↓"} {abs(delta):.1f}%</div>'
    
    html = f"""
    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <div style="color: #666; font-size: 14px; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 28px; font-weight: bold; color: #333;">{value}</div>
        {delta_html}
    </div>
    """
    return html

