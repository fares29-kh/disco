"""
Advanced Process Animation Module - Disco-style with Token Speed on Arcs
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta


def create_disco_style_animation(df, max_cases=30, animation_speed=100, sla_threshold=24):
    """
    Create Disco-style animation with tokens moving along arcs at variable speeds.
    Speed depends on transition duration: fast transitions = fast tokens, slow transitions = slow tokens.
    
    Args:
        df: Event log dataframe
        max_cases: Maximum number of cases to animate
        animation_speed: Base speed of animation in milliseconds per frame
        sla_threshold: SLA threshold for arc coloring
        
    Returns:
        plotly.graph_objects.Figure: Animated figure
    """
    # Prepare data
    df_sorted = df.sort_values(['case_id', 'timestamp']).copy()
    
    # Limit to max_cases for performance
    case_ids = df_sorted['case_id'].unique()[:max_cases]
    df_anim = df_sorted[df_sorted['case_id'].isin(case_ids)].copy()
    
    # Calculate transition durations
    df_anim['next_activity'] = df_anim.groupby('case_id')['activity'].shift(-1)
    df_anim['next_timestamp'] = df_anim.groupby('case_id')['timestamp'].shift(-1)
    df_anim['transition_duration'] = (df_anim['next_timestamp'] - df_anim['timestamp']).dt.total_seconds() / 3600
    
    # Create graph for layout
    G = nx.DiGraph()
    edge_durations = {}
    edge_frequencies = {}
    
    for _, row in df_anim.iterrows():
        if row['activity'] not in G.nodes():
            G.add_node(row['activity'])
        
        if pd.notna(row['next_activity']):
            edge = (row['activity'], row['next_activity'])
            if edge not in G.edges():
                G.add_edge(row['activity'], row['next_activity'])
                edge_durations[edge] = []
                edge_frequencies[edge] = 0
            edge_durations[edge].append(row['transition_duration'])
            edge_frequencies[edge] += 1
    
    # Calculate average durations for edges
    avg_edge_durations = {}
    for edge, durations in edge_durations.items():
        avg_edge_durations[edge] = np.mean([d for d in durations if pd.notna(d)])
    
    # Calculate hierarchical layout (like Disco)
    try:
        # Try to create hierarchical layout
        for layer, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                G.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(G, subset_key="layer", align='horizontal')
        
        # Rotate 90 degrees (vertical layout)
        pos = {node: (y, -x) for node, (x, y) in pos.items()}
    except:
        # Fallback to spring layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        except:
            pos = nx.circular_layout(G)
    
    # Create base graph (nodes and edges)
    edge_traces = []
    
    # Add edges with colors based on duration
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        avg_duration = avg_edge_durations.get(edge, 0)
        
        # Determine color based on SLA
        if avg_duration > sla_threshold:
            edge_color = '#E74C3C'  # Red
        elif avg_duration > sla_threshold * 0.5:
            edge_color = '#F39C12'  # Orange
        else:
            edge_color = '#2ECC71'  # Green
        
        # Edge width based on frequency
        freq = edge_frequencies.get(edge, 1)
        max_freq = max(edge_frequencies.values()) if edge_frequencies else 1
        edge_width = max(2, min(freq / max_freq * 10, 10))
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=edge_width, color=edge_color),
            hoverinfo='text',
            text=f"{edge[0]} → {edge[1]}<br>Avg: {avg_duration:.1f}h<br>Freq: {freq}",
            showlegend=False,
            opacity=0.6
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
    
    # Prepare animation frames with tokens moving on arcs
    frames = []
    
    # Get time range
    min_time = df_anim['timestamp'].min()
    max_time = df_anim['timestamp'].max()
    time_range = (max_time - min_time).total_seconds()
    
    # Create frames
    num_frames = min(150, int(time_range / 60))  # Max 150 frames
    if num_frames < 20:
        num_frames = 50
    
    for frame_idx in range(num_frames):
        # Calculate current time
        progress = frame_idx / num_frames
        current_time = min_time + timedelta(seconds=time_range * progress)
        
        # Track tokens and their positions
        token_x = []
        token_y = []
        token_colors = []
        token_text = []
        
        # For each case, find where it is at this time
        for case_id in case_ids:
            case_events = df_anim[df_anim['case_id'] == case_id].copy()
            
            # Find events before current time
            events_so_far = case_events[case_events['timestamp'] <= current_time]
            
            if len(events_so_far) == 0:
                continue
            
            last_event = events_so_far.iloc[-1]
            current_activity = last_event['activity']
            
            # Check if in transition
            if pd.notna(last_event['next_activity']):
                next_activity = last_event['next_activity']
                transition_start = last_event['timestamp']
                transition_end = last_event['next_timestamp']
                
                # If we're between these two times, token is on the arc
                if pd.notna(transition_end) and transition_start <= current_time <= transition_end:
                    # Calculate position on arc
                    transition_duration = (transition_end - transition_start).total_seconds()
                    time_elapsed = (current_time - transition_start).total_seconds()
                    
                    if transition_duration > 0:
                        # Progress along arc (0 to 1)
                        arc_progress = time_elapsed / transition_duration
                        
                        # Position between source and target
                        x0, y0 = pos[current_activity]
                        x1, y1 = pos[next_activity]
                        
                        # Interpolate position
                        token_x_pos = x0 + (x1 - x0) * arc_progress
                        token_y_pos = y0 + (y1 - y0) * arc_progress
                        
                        token_x.append(token_x_pos)
                        token_y.append(token_y_pos)
                        
                        # Color by priority
                        priority = last_event.get('priority', 'Low')
                        color_map = {
                            'Critical': '#E74C3C',
                            'High': '#E67E22',
                            'Medium': '#F39C12',
                            'Low': '#2ECC71'
                        }
                        token_colors.append(color_map.get(priority, '#95A5A6'))
                        token_text.append(f"{case_id}<br>{current_activity}→{next_activity}<br>{arc_progress*100:.0f}%")
                    else:
                        # Instant transition, place at target
                        x, y = pos[next_activity]
                        token_x.append(x)
                        token_y.append(y)
                        
                        priority = last_event.get('priority', 'Low')
                        color_map = {
                            'Critical': '#E74C3C',
                            'High': '#E67E22',
                            'Medium': '#F39C12',
                            'Low': '#2ECC71'
                        }
                        token_colors.append(color_map.get(priority, '#95A5A6'))
                        token_text.append(f"{case_id}<br>{next_activity}")
                else:
                    # Token is at current activity
                    if current_activity in pos:
                        x, y = pos[current_activity]
                        # Add small random offset
                        x += np.random.uniform(-0.03, 0.03)
                        y += np.random.uniform(-0.03, 0.03)
                        
                        token_x.append(x)
                        token_y.append(y)
                        
                        priority = last_event.get('priority', 'Low')
                        color_map = {
                            'Critical': '#E74C3C',
                            'High': '#E67E22',
                            'Medium': '#F39C12',
                            'Low': '#2ECC71'
                        }
                        token_colors.append(color_map.get(priority, '#95A5A6'))
                        token_text.append(f"{case_id}<br>{current_activity}")
            else:
                # Token is at final activity
                if current_activity in pos:
                    x, y = pos[current_activity]
                    x += np.random.uniform(-0.03, 0.03)
                    y += np.random.uniform(-0.03, 0.03)
                    
                    token_x.append(x)
                    token_y.append(y)
                    
                    priority = last_event.get('priority', 'Low')
                    color_map = {
                        'Critical': '#E74C3C',
                        'High': '#E67E22',
                        'Medium': '#F39C12',
                        'Low': '#2ECC71'
                    }
                    token_colors.append(color_map.get(priority, '#95A5A6'))
                    token_text.append(f"{case_id}<br>{current_activity}")
        
        # Create token trace
        token_trace = go.Scatter(
            x=token_x,
            y=token_y,
            mode='markers',
            marker=dict(
                size=15,
                color=token_colors,
                line=dict(width=2, color='white'),
                opacity=0.9
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
                title=f"🎬 Disco-Style Animation - {current_time.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                      f"Active Tokens: {len(token_x)} / {len(case_ids)}"
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
            'text': f"🎬 Disco-Style Process Animation - {len(case_ids)} Cases<br>"
                    "<sub>⚡ Vitesse des tokens = Durée des transitions (Rapide=Court, Lent=Long)</sub>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#F8F9FA',
        paper_bgcolor='white',
        hovermode='closest',
        height=700
    )
    
    return fig

