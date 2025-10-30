"""
Process Animation Module - Disco-style Process Mining Animation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta


def create_process_animation(df, max_cases=50, animation_speed=100):
    """
    Create animated process map showing cases flowing through the process.
    
    Args:
        df: Event log dataframe
        max_cases: Maximum number of cases to animate
        animation_speed: Speed of animation in milliseconds per frame
        
    Returns:
        plotly.graph_objects.Figure: Animated figure
    """
    # Prepare data
    df_sorted = df.sort_values(['case_id', 'timestamp']).copy()
    
    # Limit to max_cases for performance
    case_ids = df_sorted['case_id'].unique()[:max_cases]
    df_anim = df_sorted[df_sorted['case_id'].isin(case_ids)].copy()
    
    # Create graph for layout
    G = nx.DiGraph()
    for _, row in df_anim.iterrows():
        if row['activity'] not in G.nodes():
            G.add_node(row['activity'])
    
    # Add edges
    for case_id in case_ids:
        case_data = df_anim[df_anim['case_id'] == case_id]
        activities = case_data['activity'].tolist()
        for i in range(len(activities) - 1):
            if not G.has_edge(activities[i], activities[i + 1]):
                G.add_edge(activities[i], activities[i + 1])
    
    # Calculate layout
    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create base graph (nodes and edges)
    edge_traces = []
    
    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color='#BDC3C7'),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Add nodes
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='middle center',
        textfont=dict(size=10, color='white', family='Arial Black'),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=50,
            color='#3498DB',
            line=dict(width=3, color='white')
        ),
        showlegend=False
    )
    
    # Prepare animation frames
    frames = []
    
    # Get time range
    min_time = df_anim['timestamp'].min()
    max_time = df_anim['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()
    
    # Create frames (one per time step)
    num_frames = min(100, int(time_range / 60))  # Max 100 frames, 1 frame per minute
    
    if num_frames < 10:
        num_frames = 50  # Minimum frames for smooth animation
    
    for frame_idx in range(num_frames):
        # Calculate current time
        progress = frame_idx / num_frames
        current_time = min_time + timedelta(seconds=time_range * progress)
        
        # Get active cases at this time
        active_data = df_anim[df_anim['timestamp'] <= current_time]
        
        # Find position of each case
        token_x = []
        token_y = []
        token_colors = []
        token_text = []
        
        for case_id in case_ids:
            case_events = active_data[active_data['case_id'] == case_id]
            
            if len(case_events) > 0:
                # Get last activity for this case
                last_event = case_events.iloc[-1]
                activity = last_event['activity']
                
                if activity in pos:
                    x, y = pos[activity]
                    
                    # Add some random offset to avoid overlap
                    offset = 0.05
                    x += np.random.uniform(-offset, offset)
                    y += np.random.uniform(-offset, offset)
                    
                    token_x.append(x)
                    token_y.append(y)
                    
                    # Color based on priority
                    if 'priority' in last_event:
                        priority = last_event['priority']
                        if priority == 'Critical':
                            color = '#E74C3C'
                        elif priority == 'High':
                            color = '#E67E22'
                        elif priority == 'Medium':
                            color = '#F39C12'
                        else:
                            color = '#2ECC71'
                    else:
                        color = '#95A5A6'
                    
                    token_colors.append(color)
                    token_text.append(f"Case: {case_id}<br>Activity: {activity}")
        
        # Create token trace
        token_trace = go.Scatter(
            x=token_x,
            y=token_y,
            mode='markers',
            marker=dict(
                size=12,
                color=token_colors,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            hoverinfo='text',
            hovertext=token_text,
            showlegend=False
        )
        
        # Create frame
        frame_data = edge_traces + [node_trace, token_trace]
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=f"Process Animation - {current_time.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                      f"Active Cases: {len(token_x)} / {len(case_ids)}"
            )
        ))
    
    # Create initial figure
    initial_data = edge_traces + [node_trace]
    
    fig = go.Figure(
        data=initial_data,
        frames=frames
    )
    
    # Add play/pause buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='▶️ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': animation_speed, 'redraw': True},
                            'fromcurrent': True,
                            'mode': 'immediate',
                            'transition': {'duration': 50}
                        }]
                    ),
                    dict(
                        label='⏸️ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏮️ Reset',
                        method='animate',
                        args=[[frames[0].name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ],
                x=0.1,
                y=1.15,
                xanchor='left',
                yanchor='top'
            )
        ],
        sliders=[{
            'active': 0,
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
            ],
            'x': 0.1,
            'len': 0.85,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }],
        title={
            'text': f"🎬 Process Animation - {len(case_ids)} Cases",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='white',
        hovermode='closest',
        height=700
    )
    
    return fig


def create_token_replay(df, selected_case_id=None, animation_speed=500):
    """
    Create token replay animation for a single case or all cases.
    
    Args:
        df: Event log dataframe
        selected_case_id: Case ID to replay (None for all)
        animation_speed: Speed in milliseconds
        
    Returns:
        plotly.graph_objects.Figure
    """
    if selected_case_id:
        df_replay = df[df['case_id'] == selected_case_id].copy()
        title = f"Token Replay - Case: {selected_case_id}"
    else:
        # Take first 20 cases
        case_ids = df['case_id'].unique()[:20]
        df_replay = df[df['case_id'].isin(case_ids)].copy()
        title = f"Token Replay - {len(case_ids)} Cases"
    
    df_replay = df_replay.sort_values(['case_id', 'timestamp'])
    
    # Create graph
    activities = df_replay['activity'].unique()
    G = nx.DiGraph()
    
    for activity in activities:
        G.add_node(activity)
    
    # Add edges
    for case_id in df_replay['case_id'].unique():
        case_data = df_replay[df_replay['case_id'] == case_id]
        acts = case_data['activity'].tolist()
        for i in range(len(acts) - 1):
            if not G.has_edge(acts[i], acts[i + 1]):
                G.add_edge(acts[i], acts[i + 1])
    
    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)
    
    # Create base graph
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=3, color='#BDC3C7'),
        hoverinfo='none',
        showlegend=False
    )
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='middle center',
        textfont=dict(size=11, color='black', family='Arial Black'),
        marker=dict(size=60, color='#3498DB', line=dict(width=3, color='white')),
        showlegend=False
    )
    
    # Create frames for animation
    frames = []
    max_events = len(df_replay)
    
    for event_idx in range(max_events):
        current_events = df_replay.iloc[:event_idx + 1]
        
        # Find current position of each token
        token_x = []
        token_y = []
        token_colors = []
        token_sizes = []
        token_text = []
        
        for case_id in current_events['case_id'].unique():
            case_events = current_events[current_events['case_id'] == case_id]
            last_event = case_events.iloc[-1]
            activity = last_event['activity']
            
            if activity in pos:
                x, y = pos[activity]
                # Small random offset
                x += np.random.uniform(-0.03, 0.03)
                y += np.random.uniform(-0.03, 0.03)
                
                token_x.append(x)
                token_y.append(y)
                
                # Color by priority
                priority = last_event.get('priority', 'Low')
                color_map = {
                    'Critical': '#E74C3C',
                    'High': '#E67E22',
                    'Medium': '#F39C12',
                    'Low': '#2ECC71'
                }
                token_colors.append(color_map.get(priority, '#95A5A6'))
                token_sizes.append(15)
                token_text.append(f"{case_id}<br>{activity}")
        
        token_trace = go.Scatter(
            x=token_x, y=token_y,
            mode='markers',
            marker=dict(
                size=token_sizes,
                color=token_colors,
                line=dict(width=2, color='white')
            ),
            hovertext=token_text,
            hoverinfo='text',
            showlegend=False
        )
        
        frames.append(go.Frame(
            data=[edge_trace, node_trace, token_trace],
            name=str(event_idx)
        ))
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        frames=frames
    )
    
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                dict(label='▶️ Play', method='animate',
                     args=[None, {'frame': {'duration': animation_speed, 'redraw': True},
                                  'fromcurrent': True}]),
                dict(label='⏸️ Pause', method='animate',
                     args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                    'mode': 'immediate'}]),
                dict(label='⏮️ Reset', method='animate',
                     args=[[frames[0].name], {'frame': {'duration': 0, 'redraw': True},
                                               'mode': 'immediate'}])
            ],
            'x': 0.1, 'y': 1.12
        }],
        sliders=[{
            'active': 0,
            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                           'mode': 'immediate'}],
                       'label': str(i), 'method': 'animate'}
                      for i, f in enumerate(frames)],
            'x': 0.1, 'len': 0.85, 'y': 0
        }],
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#F8F9FA',
        height=700
    )
    
    return fig

