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


def plot_process_map(dfg_data, width=1200, height=600):
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
    
    # Create node trace with better styling and durations
    node_x = []
    node_y = []
    node_text = []
    node_labels = []
    node_colors = []
    node_sizes = []
    node_annotations = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Get node duration data
        node_info = nodes_data.get(node, {'duration_hours': 0, 'color': 'blue', 'status': 'unknown'})
        duration_hours = node_info['duration_hours']
        node_color_key = node_info['color']
        
        # Format node label with duration
        if duration_hours < 1:
            duration_text = f"{duration_hours * 60:.0f}min"
        elif duration_hours < 24:
            duration_text = f"{duration_hours:.1f}h"
        else:
            days = duration_hours / 24
            duration_text = f"{days:.1f}d"
        
        # Node label: Activity name
        node_labels.append(node)
        
        # Calculate node statistics
        in_freq = sum([e['frequency'] for e in edges if e['to'] == node])
        out_freq = sum([e['frequency'] for e in edges if e['from'] == node])
        total_freq = in_freq + out_freq
        
        # Node size based on frequency
        max_total = max([
            sum([e['frequency'] for e in edges if e['to'] == n or e['from'] == n])
            for n in G.nodes()
        ])
        node_size = max(40, min(total_freq / max_total * 70, 90))
        node_sizes.append(node_size)
        
        # Node color: RED if > 24h, BLUE otherwise
        if node_color_key == 'red':
            color_hex = '#E74C3C'  # Rouge
        else:
            color_hex = '#3498DB'  # Bleu
        node_colors.append(color_hex)
        
        # Hover text
        status_text = "⚠️ CRITICAL" if node_color_key == 'red' else "✅ Normal"
        node_text.append(
            f"<b>{node}</b><br>"
            f"Avg Duration: <b>{duration_text}</b><br>"
            f"Status: {status_text}<br>"
            f"In: {in_freq} | Out: {out_freq}<br>"
            f"Total: {total_freq}"
        )
        
        # Add duration annotation below node
        if duration_hours > 0:
            node_annotations.append(
                dict(
                    x=x,
                    y=y - 0.08,  # Below the node
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
            )
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition='middle center',
        textfont=dict(
            size=10,
            color='black',
            family='Arial Black'
        ),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=3, color='white'),
            opacity=0.9
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Add duration annotations for edges
    for annotation in edge_annotations:
        fig.add_annotation(annotation)
    
    # Add duration annotations for nodes
    for annotation in node_annotations:
        fig.add_annotation(annotation)
    
    fig.update_layout(
        title={
            'text': "Process Map - Workflow with Duration",
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

