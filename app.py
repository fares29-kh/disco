"""
Process Mining Dashboard with AI - Bug Workflow Analysis
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import traceback

# Import utility modules
from utils.data_loader import (
    load_and_validate_csv, 
    apply_filters, 
    get_filter_options,
    convert_to_pm4py_log
)
from utils.metrics import (
    calculate_kpis,
    calculate_case_durations,
    identify_bottlenecks,
    calculate_heatmap_data,
    calculate_variant_analysis,
    calculate_category_statistics
)
from utils.process_mining import (
    compute_dfg_with_colors,
    get_process_variants,
    analyze_loops,
    detect_parallel_activities
)
from utils.visualizations import (
    plot_process_map,
    plot_heatmap,
    plot_duration_distribution,
    plot_timeline,
    plot_category_comparison,
    plot_activity_frequency,
    plot_temporal_analysis,
    plot_priority_distribution,
    plot_variant_analysis
)

# Page configuration
st.set_page_config(
    page_title="Process Mining Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="stDataFrame"] {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.title("🔍 Process Mining & AI Dashboard")
st.markdown("**Intelligent Bug Workflow Analysis** - Identify bottlenecks, optimize processes, and predict SLA risks")
st.divider()

# Sidebar for file upload and filters
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload Event Log (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file containing bug event logs with columns: case_id, activity, timestamp, category, priority, severity"
    )
    
    st.divider()
    
    if uploaded_file is not None:
        st.header("⚙️ Filters")
        
        # Initialize session state for filters if not exists
        if 'df_original' not in st.session_state:
            st.session_state.df_original = None
        if 'filter_reset' not in st.session_state:
            st.session_state.filter_reset = False

# Main content
if uploaded_file is None:
    # Welcome screen
    st.info("👆 Please upload a CSV or Excel file to begin the analysis")
    
    st.markdown("""
    ### 📋 Required File Format (CSV or Excel)
    
    Your file should contain the following columns:
    
    | Column | Alternative Names | Description | Example |
    |--------|-------------------|-------------|---------|
    | `case_id` | `case:concept:name` | Unique bug identifier | BUG-001, BUG-002 |
    | `activity` | `concept:name` | Process step | Open, Analyze, Fix, Test, Close |
    | `timestamp` | `time:timestamp` | Date and time | 2024-01-01 10:30:00 |
    | `category` | `Category` | Bug category | Backend, Frontend, Database |
    | `priority` | `Priority` | Bug priority | Critical, High, Medium, Low |
    | `severity` | `Severity` | Bug severity | Blocker, Major, Minor, Trivial |
    
    **Note:** The application automatically recognizes both standard and alternative column names.
    
    ### 🎯 Features
    
    - **KPI Dashboard**: Track key metrics like resolution time, SLA compliance, and reopening rates
    - **Process Map**: Visualize workflow with color-coded bottlenecks
    - **Heatmap Analysis**: Identify slow activities by category
    - **Bottleneck Detection**: Automatically find process inefficiencies
    - **Variant Analysis**: Discover common paths through your workflow
    - **Temporal Analysis**: Track trends over time
    - **AI Insights**: Get intelligent recommendations (coming soon)
    
    ### 🚀 Quick Start
    
    1. Upload your event log CSV file
    2. Apply filters to focus on specific categories, priorities, or time periods
    3. Explore interactive visualizations and insights
    4. Export reports and recommendations
    """)
    
    # Sample data download
    st.markdown("### 📥 Need Sample Data?")
    
    col_sample1, col_sample2 = st.columns(2)
    
    with col_sample1:
        if st.button("Generate Sample CSV"):
            # Generate sample data
            sample_data = []
            activities = ['Open', 'Analyze', 'Assign', 'Fix', 'Test', 'Review', 'Close']
            categories = ['Backend', 'Frontend', 'Database', 'API', 'UI/UX']
            priorities = ['Critical', 'High', 'Medium', 'Low']
            severities = ['Blocker', 'Major', 'Minor', 'Trivial']
            
            start_date = datetime.now() - timedelta(days=30)
            
            for case_num in range(1, 51):  # 50 sample bugs
                case_id = f"BUG-{case_num:03d}"
                current_time = start_date + timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24))
                category = np.random.choice(categories)
                priority = np.random.choice(priorities)
                severity = np.random.choice(severities)
                
                # Generate activities for this case
                num_activities = np.random.randint(4, len(activities) + 1)
                case_activities = activities[:num_activities]
                
                for activity in case_activities:
                    sample_data.append({
                        'case_id': case_id,
                        'activity': activity,
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'category': category,
                        'priority': priority,
                        'severity': severity
                    })
                    # Add random time between activities
                    current_time += timedelta(hours=np.random.exponential(5))
            
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv,
                file_name="sample_bug_events.csv",
                mime="text/csv"
            )
            st.success("✅ Sample CSV generated!")
    
    with col_sample2:
        if st.button("Generate Sample Excel"):
            # Generate sample data (same as CSV)
            sample_data = []
            activities = ['Open', 'Analyze', 'Assign', 'Fix', 'Test', 'Review', 'Close']
            categories = ['Backend', 'Frontend', 'Database', 'API', 'UI/UX']
            priorities = ['Critical', 'High', 'Medium', 'Low']
            severities = ['Blocker', 'Major', 'Minor', 'Trivial']
            
            start_date = datetime.now() - timedelta(days=30)
            
            for case_num in range(1, 51):  # 50 sample bugs
                case_id = f"BUG-{case_num:03d}"
                current_time = start_date + timedelta(days=np.random.randint(0, 30), hours=np.random.randint(0, 24))
                category = np.random.choice(categories)
                priority = np.random.choice(priorities)
                severity = np.random.choice(severities)
                
                # Generate activities for this case
                num_activities = np.random.randint(4, len(activities) + 1)
                case_activities = activities[:num_activities]
                
                for activity in case_activities:
                    sample_data.append({
                        'case_id': case_id,
                        'activity': activity,
                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'category': category,
                        'priority': priority,
                        'severity': severity
                    })
                    # Add random time between activities
                    current_time += timedelta(hours=np.random.exponential(5))
            
            sample_df = pd.DataFrame(sample_data)
            
            # Convert to Excel
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                sample_df.to_excel(writer, index=False, sheet_name='Bug Events')
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download Sample Excel",
                data=excel_data,
                file_name="sample_bug_events.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            st.success("✅ Sample Excel generated!")

else:
    # Load data
    try:
        with st.spinner("Loading and validating data..."):
            df = load_and_validate_csv(uploaded_file)
            st.session_state.df_original = df
            
            st.sidebar.success(f"✅ Loaded {len(df)} events from {df['case_id'].nunique()} cases")
            
            # Get filter options
            filter_options = get_filter_options(df)
            
            # Category filter
            selected_categories = st.sidebar.multiselect(
                "Filter by Category",
                options=filter_options['categories'],
                default=filter_options['categories'],
                key='category_filter'
            )
            
            # Priority filter
            selected_priorities = st.sidebar.multiselect(
                "Filter by Priority",
                options=filter_options['priorities'],
                default=filter_options['priorities'],
                key='priority_filter'
            )
            
            # Severity filter
            selected_severities = st.sidebar.multiselect(
                "Filter by Severity",
                options=filter_options['severities'],
                default=filter_options['severities'],
                key='severity_filter'
            )
            
            # Date range filter
            st.sidebar.subheader("Date Range")
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=(filter_options['min_date'], filter_options['max_date']),
                min_value=filter_options['min_date'],
                max_value=filter_options['max_date'],
                key='date_filter'
            )
            
            # SLA threshold slider
            st.sidebar.subheader("SLA Configuration")
            sla_threshold = st.sidebar.slider(
                "SLA Threshold (hours)",
                min_value=1,
                max_value=168,  # 1 week
                value=24,
                help="Define the maximum acceptable resolution time"
            )
            
            # Reset filters button
            if st.sidebar.button("🔄 Reset All Filters"):
                st.session_state.filter_reset = True
                st.rerun()
            
            # Apply filters
            df_filtered = apply_filters(
                df,
                categories=selected_categories,
                priorities=selected_priorities,
                severities=selected_severities,
                date_range=date_range if len(date_range) == 2 else None
            )
            
            if df_filtered.empty:
                st.error("❌ No data matches the selected filters. Please adjust your filter criteria.")
                st.stop()
            
            st.sidebar.info(f"📊 Filtered: {df_filtered['case_id'].nunique()} cases")
    
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Calculate metrics
    try:
        with st.spinner("Calculating metrics..."):
            kpis = calculate_kpis(df_filtered, sla_threshold)
            case_durations = calculate_case_durations(df_filtered)
    except Exception as e:
        st.error(f"❌ Error calculating metrics: {str(e)}")
        st.exception(e)
        st.stop()
    
    # Display KPIs
    st.header("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Bugs",
            value=kpis['total_bugs'],
            help="Total number of unique bug cases"
        )
    
    with col2:
        st.metric(
            label="Avg Resolution Time",
            value=f"{kpis['avg_resolution_time']:.1f}h",
            help="Average time from bug opening to closure"
        )
    
    with col3:
        st.metric(
            label="SLA Risk %",
            value=f"{kpis['sla_risk_percentage']:.1f}%",
            delta=f"-{kpis['sla_risk_percentage']:.1f}%" if kpis['sla_risk_percentage'] < 20 else None,
            delta_color="inverse",
            help=f"{kpis['sla_risk_count']} cases exceed {sla_threshold}h threshold"
        )
    
    with col4:
        st.metric(
            label="Reopenings",
            value=kpis['reopen_count'],
            delta=f"{kpis['reopen_rate']:.1f}%" if kpis['reopen_rate'] > 0 else None,
            delta_color="inverse",
            help="Number of bugs that were reopened"
        )
    
    with col5:
        st.metric(
            label="Completion Rate",
            value=f"{kpis['completion_rate']:.1f}%",
            delta=f"+{kpis['completion_rate']:.1f}%" if kpis['completion_rate'] > 80 else None,
            help=f"{kpis['closed_count']} cases closed"
        )
    
    # Slowest bug highlight
    st.info(f"🐌 **Slowest Bug**: {kpis['slowest_bug_id']} took {kpis['slowest_bug_duration']:.1f} hours to resolve")
    
    st.divider()
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🗺️ Process Map",
        "🔥 Heatmap & Bottlenecks",
        "📈 Distributions",
        "🔄 Variants & Loops",
        "📅 Temporal Analysis",
        "🤖 AI Predictions",
        "🎬 Animation"
    ])
    
    with tab1:
        st.header("Process Map Visualization")
        st.markdown("**Interactive process flow diagram** - 🟢 Green boxes show START nodes (entry points), 🟣 Purple boxes show END nodes (exit points)")
        
        # Option to enable animation
        col_anim1, col_anim2 = st.columns([1, 4])
        with col_anim1:
            enable_animation = st.checkbox("🎬 Activer l'animation", value=False, help="Affiche des cercles animés qui se déplacent le long des lignes. La vitesse dépend de la couleur : vert (rapide), orange (normal), rouge (lent).")
        with col_anim2:
            if enable_animation:
                st.info("💡 L'animation affiche des tokens colorés qui se déplacent le long des transitions. Cliquez sur '▶️ Play' après la génération pour démarrer l'animation.")
        
        try:
            with st.spinner("Generating process map..."):
                dfg_data = compute_dfg_with_colors(df_filtered, sla_threshold)
                fig_process = plot_process_map(dfg_data, animated=enable_animation)
                st.plotly_chart(fig_process, use_container_width=True)
            
            # Legend with better explanations
            st.markdown("---")
            st.markdown("### 📊 Légende")
            
            st.markdown("#### 🔗 Arcs (Transitions entre activités)")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("🟢 **Vert**: < 50% du SLA")
                st.caption(f"Durée < {sla_threshold * 0.5:.1f}h")
            with col2:
                st.markdown("🟠 **Orange**: 50-100% du SLA")
                st.caption(f"Durée entre {sla_threshold * 0.5:.1f}h et {sla_threshold:.1f}h")
            with col3:
                st.markdown("🔴 **Rouge**: > SLA (Critique)")
                st.caption(f"Durée > {sla_threshold:.1f}h")
            
            st.markdown("#### 🔵 Nœuds (Activités/Tâches)")
            
            # Highlight start/end nodes prominently
            st.info("💡 **Points Clés**: 🟢 **VERT = DÉPART** (première activité) | 🟣 **VIOLET = FIN** (dernière activité)")
            
            col4, col5, col6, col7 = st.columns(4)
            with col4:
                st.markdown("🟢 **Vert + 🚀**: Nœud de DÉPART")
                st.caption("Point d'entrée du processus")
            with col5:
                st.markdown("🟣 **Violet + 🏁**: Nœud de FIN")
                st.caption("Point de sortie du processus")
            with col6:
                st.markdown("🔵 **Bleu**: Durée normale")
                st.caption("Temps moyen ≤ 24h")
            with col7:
                st.markdown("🔴 **Rouge**: Durée critique")
                st.caption("Temps moyen > 24h")
            
            st.info(f"ℹ️ **Seuil SLA pour arcs**: {sla_threshold}h (ajustable) | **Seuil pour nœuds**: 24h (fixe)")
            
            # Additional explanations
            with st.expander("💡 Comment lire le Process Map"):
                st.markdown("""
                **Nœuds (rectangles)** :
                - Représentent les activités/tâches du processus
                - **Identification des Points Clés** :
                  - 🟢 **VERT avec badge "🚀 START"** : Nœud de DÉPART
                    * Point d'entrée du processus
                    * Première activité effectuée
                    * Identifiable par la couleur verte distinctive
                  - 🟣 **VIOLET avec badge "🏁 END"** : Nœud de FIN
                    * Point de sortie du processus
                    * Dernière activité effectuée
                    * Identifiable par la couleur violette distinctive
                - **Autres Couleurs** :
                  - 🔵 **Bleu** : Activité intermédiaire avec temps moyen ≤ 24h (normal)
                  - 🔴 **Rouge** : Activité problématique avec temps moyen > 24h (critique)
                - **Caractéristiques visuelles** :
                  - Les nœuds START/END ont une bordure **plus épaisse** (5px vs 3px)
                  - Les nœuds START/END ont des badges supplémentaires au-dessus
                  - Police plus grande pour START/END (12pt vs 11pt)
                - **Label en dessous** : Durée moyenne de traitement
                
                **Arcs (flèches)** :
                - Représentent les transitions entre activités
                - **Épaisseur** : Proportionnelle à la fréquence (nombre de passages)
                - **Couleur** : Performance par rapport au SLA
                  - 🟢 **Vert** : < 50% du SLA (performant)
                  - 🟠 **Orange** : 50-100% du SLA (attention)
                  - 🔴 **Rouge** : > SLA (critique)
                - **Labels** : Durée moyenne de transition + fréquence (entre parenthèses)
                
                **Durées affichées** :
                - Format automatique : minutes (< 1h), heures (< 24h), jours (≥ 24h)
                
                **Comment identifier les bottlenecks** :
                1. **Nœuds rouges** : Activités qui prennent > 24h en moyenne
                2. **Arcs rouges** : Transitions qui dépassent le SLA
                3. **Arcs épais + rouges** : Haute fréquence ET lenteur = goulot critique
                
                **Animation (si activée)** :
                - 🎬 **Tokens animés** : Des cercles colorés se déplacent le long des transitions
                - **Vitesse** : Dépend de la couleur de l'arc
                  - 🟢 **Vert** : Tokens rapides (transition performante)
                  - 🟠 **Orange** : Vitesse normale
                  - 🔴 **Rouge** : Tokens lents (goulot d'étranglement)
                - **Utilisation** : Cliquez sur "▶️ Play" après avoir activé l'animation
                - **Objectif** : Visualiser le flux réel et identifier visuellement les ralentissements
                
                **Actions recommandées** :
                - 🔴 **Nœuds/Arcs rouges** : Priorité maximale, action immédiate
                - ⚠️ **Arcs oranges** : À surveiller, optimisation possible
                - ✅ **Arcs verts** : Performants, à maintenir
                """)
        
        except Exception as e:
            st.error(f"Error generating process map: {str(e)}")
            st.exception(e)
    
    with tab2:
        st.header("Heatmap & Bottleneck Analysis")
        
        # Heatmap
        try:
            with st.spinner("Generating heatmap..."):
                heatmap_data = calculate_heatmap_data(df_filtered)
                if not heatmap_data.empty:
                    fig_heatmap = plot_heatmap(heatmap_data)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.warning("Not enough data to generate heatmap")
        except Exception as e:
            st.error(f"Error generating heatmap: {str(e)}")
        
        st.divider()
        
        # Bottlenecks table
        st.subheader("⚠️ Top Bottlenecks")
        try:
            bottlenecks = identify_bottlenecks(df_filtered, sla_threshold, top_n=10)
            
            if not bottlenecks.empty:
                # Display as styled dataframe
                st.dataframe(
                    bottlenecks[['from_activity', 'to_activity', 'avg_duration_hours', 'frequency', 'sla_status']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "from_activity": "From Activity",
                        "to_activity": "To Activity",
                        "avg_duration_hours": st.column_config.NumberColumn(
                            "Avg Duration (hours)",
                            format="%.2f"
                        ),
                        "frequency": "Frequency",
                        "sla_status": "SLA Status"
                    }
                )
            else:
                st.info("No bottlenecks detected")
        except Exception as e:
            st.error(f"Error identifying bottlenecks: {str(e)}")
    
    with tab3:
        st.header("Statistical Distributions")
        
        # Duration distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Case Duration Distribution")
            try:
                fig_dist = plot_duration_distribution(case_durations, sla_threshold)
                st.plotly_chart(fig_dist, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        with col2:
            st.subheader("Priority Distribution")
            try:
                fig_priority = plot_priority_distribution(df_filtered)
                st.plotly_chart(fig_priority, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Activity frequency
        st.subheader("Activity Frequency")
        try:
            fig_activity = plot_activity_frequency(df_filtered)
            st.plotly_chart(fig_activity, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        # Category comparison
        st.subheader("Category Performance")
        try:
            category_stats = calculate_category_statistics(df_filtered)
            fig_category = plot_category_comparison(category_stats)
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Show table
            with st.expander("View Category Statistics Table"):
                st.dataframe(category_stats, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab4:
        st.header("Process Variants & Loops")
        
        # Variant analysis
        # Initialize variant_df to avoid NameError
        variant_df = pd.DataFrame()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top Process Variants")
            try:
                variant_df = get_process_variants(df_filtered, top_n=10)
                if not variant_df.empty:
                    fig_variants = plot_variant_analysis(variant_df)
                    st.plotly_chart(fig_variants, use_container_width=True)
                else:
                    st.info("No variant data available")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        with col2:
            st.subheader("Variant Statistics")
            try:
                if not variant_df.empty:
                    st.metric("Total Variants", len(variant_df))
                    st.metric("Most Common", f"{variant_df.iloc[0]['percentage']:.1f}%")
                    
                    with st.expander("View All Variants"):
                        st.dataframe(variant_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No variant statistics available")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Loop detection
        st.subheader("🔄 Detected Loops (Rework)")
        try:
            loops = analyze_loops(df_filtered)
            if not loops.empty:
                st.dataframe(loops, use_container_width=True, hide_index=True)
                st.warning(f"Found {len(loops)} activities with repetitions (potential rework)")
            else:
                st.success("✅ No loops detected - workflow is linear")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab5:
        st.header("Temporal Analysis")
        
        # Timeline
        st.subheader("Critical Cases Timeline")
        try:
            fig_timeline = plot_timeline(df_filtered, sla_threshold, max_cases=30)
            st.plotly_chart(fig_timeline, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Cases over time
        st.subheader("Case Volume Over Time")
        try:
            fig_temporal = plot_temporal_analysis(df_filtered)
            st.plotly_chart(fig_temporal, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with tab6:
        st.header("🤖 AI-Powered Predictive Analysis")
        st.markdown("Predict bug resolution time and complexity using Machine Learning")
        
        # Import ML modules
        from utils.ml_models import BugDurationPredictor, train_model_cached, compare_models
        from utils.feature_engineering import (
            prepare_features_for_prediction,
            calculate_complexity_score,
            extract_features_from_log
        )
        
        # Sub-tabs for different AI features
        ai_tab1, ai_tab2, ai_tab3, ai_tab4 = st.tabs([
            "📊 Model Training & Evaluation",
            "🔮 Predict New Bug Instance",
            "🏆 Category Prioritization",
            "📉 Overall Process Performance"
        ])
        
        with ai_tab1:
            # Header section with better styling
            st.markdown("### 🎓 Train ML Model on Historical Data")
            st.markdown("---")
            
            # Check if enough data
            if df_filtered['case_id'].nunique() < 10:
                st.warning(f"⚠️ **Insufficient Data**: Need at least 10 completed bugs to train a model. Current: {df_filtered['case_id'].nunique()} cases")
            else:
                # Model selection section with better layout
                st.markdown("#### 📋 Configuration")
                
                col1, col2, col3 = st.columns([3, 2, 2])
                
                with col1:
                    model_type = st.selectbox(
                        "**Select Model Type**",
                        options=['random_forest', 'gradient_boosting', 'linear'],
                        format_func=lambda x: {
                            'random_forest': '🌲 Random Forest (Recommended)',
                            'gradient_boosting': '🚀 Gradient Boosting',
                            'linear': '📏 Linear Regression'
                        }[x],
                        help="Choose the machine learning algorithm to use"
                    )
                
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    train_button = st.button("🎯 Train Model", type="primary", use_container_width=True)
                
                with col3:
                    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
                    data_info = st.info(f"📊 **{df_filtered['case_id'].nunique()}** cases available", icon="ℹ️")
                
                if train_button:
                    with st.spinner("🔄 Training ML model... This may take a few moments."):
                        try:
                            # Train model
                            predictor = BugDurationPredictor(model_type=model_type)
                            metrics = predictor.train(df_filtered)
                            
                            # Store in session state
                            st.session_state.predictor = predictor
                            st.session_state.model_trained = True
                            
                            st.success("✅ **Model trained successfully!**")
                            
                            st.markdown("---")
                            
                            # Display metrics with better styling
                            st.markdown("#### 📊 Model Performance Metrics")
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                st.metric(
                                    "📉 Test MAE",
                                    f"{metrics['test_mae']:.2f}h",
                                    help="Mean Absolute Error on test set - Lower is better",
                                    delta=f"±{metrics['test_mae']:.1f}h average error"
                                )
                            
                            with metric_col2:
                                st.metric(
                                    "📊 Test RMSE",
                                    f"{metrics['test_rmse']:.2f}h",
                                    help="Root Mean Squared Error on test set - Lower is better",
                                    delta=f"{metrics['test_rmse']:.1f}h std deviation"
                                )
                            
                            with metric_col3:
                                # Color code R² score
                                r2_score = metrics['test_r2']
                                if r2_score >= 0.8:
                                    delta_color = "normal"
                                    delta_text = "Excellent"
                                elif r2_score >= 0.6:
                                    delta_color = "normal"
                                    delta_text = "Good"
                                elif r2_score >= 0.4:
                                    delta_color = "off"
                                    delta_text = "Moderate"
                                else:
                                    delta_color = "inverse"
                                    delta_text = "Needs Improvement"
                                
                                st.metric(
                                    "🎯 R² Score",
                                    f"{r2_score:.3f}",
                                    delta=delta_text,
                                    delta_color=delta_color,
                                    help="Coefficient of determination (1.0 = perfect prediction)"
                                )
                            
                            # Performance interpretation
                            st.markdown("---")
                            if r2_score >= 0.8:
                                st.success(f"🎉 **Excellent Model Performance!** Your model explains {r2_score*100:.1f}% of the variance in bug resolution times.")
                            elif r2_score >= 0.6:
                                st.info(f"✅ **Good Model Performance.** Your model explains {r2_score*100:.1f}% of the variance. Consider adding more features for better accuracy.")
                            else:
                                st.warning(f"⚠️ **Model Performance Needs Improvement.** R² score of {r2_score:.3f} suggests the model may need more training data or feature engineering.")
                            
                            # Show feature importance for tree-based models
                            if predictor.feature_importance is not None:
                                st.markdown("---")
                                st.markdown("#### 🎯 Feature Importance Analysis")
                                st.markdown("*Understanding which features are most influential in predicting bug resolution time*")
                                
                                import plotly.express as px
                                
                                # Prepare data for visualization
                                importance_df = predictor.feature_importance.head(10).copy()
                                importance_df = importance_df.sort_values('importance', ascending=True)
                                
                                # Create better styled chart
                                fig_importance = px.bar(
                                    importance_df,
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title='<b>Top 10 Most Important Features</b>',
                                    labels={
                                        'importance': '<b>Importance Score</b>',
                                        'feature': '<b>Feature</b>'
                                    },
                                    color='importance',
                                    color_continuous_scale='Blues',
                                    text='importance'
                                )
                                
                                fig_importance.update_traces(
                                    texttemplate='%{text:.3f}',
                                    textposition='outside',
                                    marker=dict(line=dict(color='white', width=1))
                                )
                                
                                fig_importance.update_layout(
                                    height=500,
                                    showlegend=False,
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    title_font_size=18,
                                    title_x=0.5,
                                    xaxis_title_font_size=14,
                                    yaxis_title_font_size=14,
                                    margin=dict(l=150, r=50, t=80, b=50)
                                )
                                
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Feature importance table
                                with st.expander("📋 View All Feature Importances"):
                                    display_importance = predictor.feature_importance.copy()
                                    display_importance['importance'] = display_importance['importance'].round(4)
                                    display_importance = display_importance.rename(columns={
                                        'feature': 'Feature',
                                        'importance': 'Importance Score'
                                    })
                                    st.dataframe(display_importance, use_container_width=True, hide_index=True)
                            
                            st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"❌ **Error training model**: {str(e)}")
                            import traceback
                            with st.expander("🔍 Show detailed error trace"):
                                st.code(traceback.format_exc())
                
                # Model comparison section
                st.markdown("---")
                st.markdown("#### 🏆 Model Comparison")
                st.markdown("*Compare different ML algorithms to find the best performing model*")
                
                if st.button("🔄 Compare All Models", type="secondary", use_container_width=False):
                    with st.spinner("🔄 Comparing all models... This may take a minute."):
                        try:
                            comparison = compare_models(df_filtered)
                            
                            # Style the comparison table
                            styled_comparison = comparison.style.format({
                                'test_mae': '{:.2f}',
                                'test_rmse': '{:.2f}',
                                'test_r2': '{:.3f}',
                                'train_mae': '{:.2f}',
                                'train_r2': '{:.3f}'
                            }).highlight_min(
                                subset=['test_mae', 'test_rmse'], 
                                color='#90EE90',
                                axis=0
                            ).highlight_max(
                                subset=['test_r2'], 
                                color='#90EE90',
                                axis=0
                            )
                            
                            st.dataframe(styled_comparison, use_container_width=True, hide_index=True)
                            
                            # Find best model
                            best_model_idx = comparison['test_r2'].idxmax()
                            best_model = comparison.loc[best_model_idx]
                            
                            st.success(f"🏆 **Best Model**: {best_model['model'].replace('_', ' ').title()} with R² = {best_model['test_r2']:.3f}")
                            
                        except Exception as e:
                            st.error(f"❌ **Error comparing models**: {str(e)}")
                            import traceback
                            with st.expander("🔍 Show detailed error"):
                                st.code(traceback.format_exc())
        
        with ai_tab2:
            st.subheader("🔮 Predict Duration for New Bug")
            
            # Check if model is trained
            if 'model_trained' not in st.session_state or not st.session_state.model_trained:
                st.info("👈 Please train a model first in the 'Model Training & Evaluation' tab")
            else:
                st.markdown("Enter the characteristics of your new bug to predict its resolution time:")
                
                # Input form
                with st.form("prediction_form"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Get unique values from data
                        categories = sorted(df_filtered['category'].unique().tolist())
                        category_input = st.selectbox("Category", categories)
                    
                    with col2:
                        priorities = sorted(df_filtered['priority'].unique().tolist())
                        priority_input = st.selectbox("Priority", priorities)
                    
                    with col3:
                        severities = sorted(df_filtered['severity'].unique().tolist())
                        severity_input = st.selectbox("Severity", severities)
                    
                    col4, col5 = st.columns(2)
                    
                    with col4:
                        start_hour_input = st.slider(
                            "Reported Hour",
                            0, 23, 9,
                            help="Hour when bug was reported (0-23)"
                        )
                    
                    with col5:
                        start_day_input = st.selectbox(
                            "Reported Day",
                            options=[0, 1, 2, 3, 4, 5, 6],
                            format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 
                                                    'Thursday', 'Friday', 'Saturday', 'Sunday'][x]
                        )
                    
                    submit_button = st.form_submit_button("🎯 Predict Duration", type="primary")
                    
                    if submit_button:
                        try:
                            # Prepare features
                            features_df = prepare_features_for_prediction(
                                category=category_input,
                                priority=priority_input,
                                severity=severity_input,
                                historical_data=extract_features_from_log(df_filtered),
                                start_hour=start_hour_input,
                                start_day_of_week=start_day_input
                            )
                            
                            # Make prediction
                            predictor = st.session_state.predictor
                            predicted_hours = predictor.predict(features_df)[0]
                            
                            # Calculate complexity score
                            hist_data = extract_features_from_log(df_filtered)
                            similar_bugs = hist_data[
                                (hist_data['category'] == category_input) &
                                (hist_data['priority'] == priority_input)
                            ]
                            
                            if len(similar_bugs) > 0:
                                hist_avg = similar_bugs['duration_hours'].mean()
                                hist_std = similar_bugs['duration_hours'].std()
                            else:
                                hist_avg = hist_data['duration_hours'].mean()
                                hist_std = hist_data['duration_hours'].std()
                            
                            complexity_score, risk_level = calculate_complexity_score(
                                predicted_hours, hist_avg, hist_std
                            )
                            
                            # Calculate process deviation
                            from utils.feature_engineering import calculate_process_deviation
                            
                            # Get similar bugs for deviation calculation
                            similar_bugs_df = df_filtered[
                                (df_filtered['category'] == category_input) &
                                (df_filtered['priority'] == priority_input)
                            ]
                            
                            if similar_bugs_df.empty:
                                similar_bugs_df = df_filtered[df_filtered['category'] == category_input]
                            
                            if similar_bugs_df.empty:
                                similar_bugs_df = df_filtered
                            
                            deviation = calculate_process_deviation(
                                df_filtered, 
                                category=category_input if not similar_bugs_df.empty else None
                            )
                            
                            # Display results
                            st.success("✅ Prediction completed!")
                            
                            st.divider()
                            
                            # Big metrics
                            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                            
                            with result_col1:
                                st.metric(
                                    "📅 Estimated Duration",
                                    f"{predicted_hours:.1f} hours",
                                    help="Predicted time to resolve this bug"
                                )
                                
                                # Convert to days if > 24h
                                if predicted_hours > 24:
                                    st.caption(f"≈ {predicted_hours/24:.1f} days")
                            
                            with result_col2:
                                # Color based on risk
                                risk_colors = {
                                    'Low': '🟢',
                                    'Medium': '🟡',
                                    'High': '🟠',
                                    'Critical': '🔴'
                                }
                                st.metric(
                                    "⚠️ Risk Level",
                                    f"{risk_colors[risk_level]} {risk_level}",
                                    help="Risk of deviating from normal process"
                                )
                            
                            with result_col3:
                                st.metric(
                                    "📊 Complexity Score",
                                    f"{complexity_score:.0f}/100",
                                    help="Higher score = more complex than average"
                                )
                            
                            with result_col4:
                                # Process deviation indicator
                                deviation_level = deviation.get('deviation_level', 'Normal')
                                deviation_colors = {
                                    'Normal': '🟢',
                                    'Moderate': '🟡',
                                    'High': '🟠',
                                    'Critical': '🔴'
                                }
                                st.metric(
                                    "🔄 Process Deviation",
                                    f"{deviation_colors.get(deviation_level, '⚪')} {deviation_level}",
                                    help="How much this case type deviates from standard process"
                                )
                                st.caption(f"Score: {deviation.get('deviation_score', 0):.0f}/100")
                            
                            # Additional insights
                            st.divider()
                            st.subheader("💡 Insights")
                            
                            col_insight1, col_insight2 = st.columns(2)
                            
                            with col_insight1:
                                if len(similar_bugs) > 0:
                                    st.info(
                                        f"📊 Based on {len(similar_bugs)} similar bugs "
                                        f"({category_input} + {priority_input}), "
                                        f"average duration is {hist_avg:.1f}h"
                                    )
                            
                            with col_insight2:
                                if deviation.get('deviation_score', 0) > 30:
                                    st.warning(
                                        f"🔄 **Process Deviation Detected** ({deviation.get('deviation_level', 'Normal')}):\n"
                                        + "\n".join([f"- {factor}" for factor in deviation.get('factors', [])])
                                    )
                                else:
                                    st.success("✅ Normal process flow expected")
                            
                            if predicted_hours > sla_threshold:
                                st.warning(
                                    f"⚠️ This bug is predicted to exceed your SLA threshold "
                                    f"of {sla_threshold}h by {predicted_hours - sla_threshold:.1f}h"
                                )
                            else:
                                st.success(
                                    f"✅ This bug is predicted to be resolved within SLA "
                                    f"({predicted_hours:.1f}h < {sla_threshold}h)"
                                )
                            
                            # Recommendations
                            st.subheader("🎯 Recommendations")
                            
                            if risk_level in ['High', 'Critical']:
                                st.markdown("""
                                - 🚨 **High Priority**: Assign experienced developers
                                - 👥 **Consider**: Pair programming or team review
                                - 📋 **Monitor**: Close tracking required
                                - 💬 **Communication**: Keep stakeholders updated
                                """)
                            elif risk_level == 'Medium':
                                st.markdown("""
                                - ⚠️ **Standard Process**: Follow normal workflow
                                - 📊 **Monitor**: Regular status updates
                                - 🔍 **Review**: Check for similar past issues
                                """)
                            else:
                                st.markdown("""
                                - ✅ **Low Risk**: Standard assignment process
                                - 🚀 **Quick Win**: Should resolve quickly
                                """)
                            
                        except Exception as e:
                            st.error(f"Error making prediction: {str(e)}")
                            import traceback
                            with st.expander("Show detailed error"):
                                st.code(traceback.format_exc())
        
        with ai_tab3:
            st.subheader("🏆 AI-Based Bug Category Prioritization")
            st.markdown("Prioritize bug categories based on their impact on process efficiency using AI analysis.")
            
            # Help section
            with st.expander("ℹ️ About Category Recommendations"):
                st.markdown("""
                ### 🎯 How it works
                
                Our AI system analyzes your bug data and provides **category-specific recommendations** based on:
                - Historical resolution times
                - SLA breach rates
                - Process deviation patterns
                - Instance frequency
                
                ### 📊 Priority Levels
                
                Categories are classified into 4 priority levels:
                - 🔴 **Critical**: Immediate action required (Security, High SLA breach)
                - 🟠 **High**: Should be addressed soon (Performance, Backend issues)
                - 🟡 **Medium**: Normal priority (UI, Testing bugs)
                - 🟢 **Low**: Can be deferred (Minor issues)
                
                ### 💡 Recommendations Include
                
                For each category, you'll get:
                1. **Insights**: Key observations about the category's behavior
                2. **Actions**: Specific steps to improve resolution efficiency
                3. **KPIs**: Metrics to monitor for continuous improvement
                
                ### 🏷️ Supported Categories
                
                The system provides tailored recommendations for:
                - **Functional Bugs**: Feature and functionality issues
                - **Performance Bugs**: Speed, latency, memory issues
                - **Testing/QA Bugs**: Quality assurance and validation
                - **GUI/UI Bugs**: Visual and interface issues
                - **Backend Bugs**: Server, API, database issues
                - **Security Bugs**: Authentication, vulnerabilities
                - **Integration Bugs**: Inter-system communication
                - **Generic**: Adaptive recommendations for other categories
                """)
            
            st.divider()
            
            if 'category' not in df_filtered.columns:
                st.warning("⚠️ No 'category' column found in the data. Category prioritization requires category information.")
            else:
                # Get predictor if available
                predictor = None
                if 'model_trained' in st.session_state and st.session_state.model_trained:
                    predictor = st.session_state.get('predictor', None)
                
                if st.button("🎯 Analyze & Prioritize Categories", type="primary"):
                    with st.spinner("Analyzing categories and calculating priority scores..."):
                        try:
                            from utils.category_prioritization import prioritize_categories
                            
                            priority_df = prioritize_categories(
                                df_filtered, 
                                sla_threshold=sla_threshold,
                                predictor=predictor
                            )
                            
                            if not priority_df.empty:
                                st.success(f"✅ Analyzed {len(priority_df)} categories!")
                                
                                # Display ranking table
                                st.subheader("📊 Category Priority Ranking")
                                
                                # Format columns for display
                                display_df = priority_df.copy()
                                # Remove "All Bugs" category if present
                                display_df = display_df[display_df['category'] != 'All Bugs'].copy()
                                display_df['priority_score'] = display_df['priority_score'].round(1)
                                display_df['predicted_resolution_time'] = display_df['predicted_resolution_time'].round(1)
                                display_df['predicted_delay_risk'] = display_df['predicted_delay_risk'].round(1)
                                display_df['deviation_score'] = display_df['deviation_score'].round(1)
                                
                                # Color code by priority score
                                def highlight_priority(row):
                                    if row['priority_score'] >= 70:
                                        return ['background-color: #ffcccc'] * len(row)
                                    elif row['priority_score'] >= 40:
                                        return ['background-color: #fff4cc'] * len(row)
                                    else:
                                        return ['background-color: #ccffcc'] * len(row)
                                
                                st.dataframe(
                                    display_df.style.apply(highlight_priority, axis=1),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Visualization
                                st.subheader("📈 Priority Score Visualization")
                                
                                import plotly.express as px
                                # Filter out "All Bugs" for visualization
                                viz_df = priority_df[priority_df['category'] != 'All Bugs'].copy()
                                fig_priority = px.bar(
                                    viz_df.head(10),
                                    x='priority_score',
                                    y='category',
                                    orientation='h',
                                    title='Top 10 Categories by Priority Score',
                                    color='priority_score',
                                    color_continuous_scale='RdYlGn_r',
                                    labels={'priority_score': 'Priority Score (0-100)', 'category': 'Category'}
                                )
                                fig_priority.update_layout(height=400)
                                st.plotly_chart(fig_priority, use_container_width=True)
                                
                                # Summary of priority levels
                                st.markdown("---")
                                st.subheader("🎯 Priority Level Summary")
                                
                                from utils.category_prioritization import get_category_recommendations
                                
                                # Count categories by priority level
                                priority_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
                                filtered_priority_df_temp = priority_df[priority_df['category'] != 'All Bugs'].copy()
                                
                                for idx, row in filtered_priority_df_temp.iterrows():
                                    metrics = {
                                        'predicted_resolution_time': row['predicted_resolution_time'],
                                        'predicted_delay_risk': row['predicted_delay_risk'],
                                        'avg_duration': row['avg_duration'],
                                        'instance_count': row['instance_count'],
                                        'deviation_score': row['deviation_score'],
                                        'priority_score': row['priority_score']
                                    }
                                    recs = get_category_recommendations(row['category'], metrics)
                                    priority_counts[recs['priority_level']] += 1
                                
                                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                with summary_col1:
                                    st.metric("🔴 Critical", priority_counts['Critical'], 
                                             help="Requires immediate attention")
                                with summary_col2:
                                    st.metric("🟠 High", priority_counts['High'],
                                             help="Should be handled soon")
                                with summary_col3:
                                    st.metric("🟡 Medium", priority_counts['Medium'],
                                             help="Normal priority")
                                with summary_col4:
                                    st.metric("🟢 Low", priority_counts['Low'],
                                             help="Can be deferred")
                                
                                if priority_counts['Critical'] > 0:
                                    st.error(f"⚠️ **{priority_counts['Critical']} Critical categories** require immediate attention!")
                                elif priority_counts['High'] > 0:
                                    st.warning(f"⚡ **{priority_counts['High']} High priority categories** should be addressed soon")
                                else:
                                    st.success("✅ No critical issues detected - All categories under control")
                                
                                st.markdown("---")
                                
                                # Detailed metrics per category with recommendations
                                st.subheader("📋 Detailed Analysis & Recommendations")
                                
                                # Import recommendation function
                                from utils.category_prioritization import get_category_recommendations
                                
                                # Filter out "All Bugs" for detailed metrics
                                filtered_priority_df = priority_df[priority_df['category'] != 'All Bugs'].copy()
                                for idx, row in filtered_priority_df.iterrows():
                                    # Get category-specific recommendations
                                    metrics = {
                                        'predicted_resolution_time': row['predicted_resolution_time'],
                                        'predicted_delay_risk': row['predicted_delay_risk'],
                                        'avg_duration': row['avg_duration'],
                                        'instance_count': row['instance_count'],
                                        'deviation_score': row['deviation_score'],
                                        'priority_score': row['priority_score']
                                    }
                                    recommendations = get_category_recommendations(row['category'], metrics)
                                    
                                    # Priority level indicator
                                    priority_colors = {
                                        'Critical': '🔴',
                                        'High': '🟠',
                                        'Medium': '🟡',
                                        'Low': '🟢'
                                    }
                                    priority_icon = priority_colors.get(recommendations['priority_level'], '⚪')
                                    
                                    with st.expander(f"{priority_icon} **{row['category']}** - Priority Score: {row['priority_score']:.1f} ({recommendations['priority_level']})"):
                                        # Metrics section
                                        st.markdown("### 📊 Key Metrics")
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Predicted Resolution Time", f"{row['predicted_resolution_time']:.1f}h")
                                            st.metric("Average Duration", f"{row['avg_duration']:.1f}h")
                                        
                                        with col2:
                                            st.metric("Delay Risk", f"{row['predicted_delay_risk']:.1f}%")
                                            st.metric("Instance Count", int(row['instance_count']))
                                        
                                        with col3:
                                            st.metric("Deviation Score", f"{row['deviation_score']:.1f}/100")
                                            st.metric("Suggested Action", row['suggested_action'])
                                        
                                        st.divider()
                                        
                                        # Insights section
                                        st.markdown("### 💡 Insights")
                                        for insight in recommendations['insights']:
                                            st.info(f"ℹ️ {insight}")
                                        
                                        # Warnings
                                        if row['predicted_delay_risk'] > 50:
                                            st.error("🚨 **High delay risk** - Many instances exceed SLA threshold. Immediate action required!")
                                        if row['deviation_score'] > 50:
                                            st.warning("⚠️ **High process deviation** - Cases deviate significantly from standard process")
                                        
                                        st.divider()
                                        
                                        # Recommendations section
                                        st.markdown("### 🎯 Recommended Actions")
                                        for action in recommendations['actions']:
                                            st.markdown(f"- {action}")
                                        
                                        st.divider()
                                        
                                        # KPIs to monitor
                                        st.markdown("### 📈 Key Performance Indicators to Monitor")
                                        kpi_cols = st.columns(len(recommendations['kpis']))
                                        for i, kpi in enumerate(recommendations['kpis']):
                                            with kpi_cols[i]:
                                                st.markdown(f"**{kpi}**")
                            else:
                                st.warning("No category data available for prioritization")
                                
                        except Exception as e:
                            st.error(f"Error analyzing categories: {str(e)}")
                            import traceback
                            with st.expander("Show detailed error"):
                                st.code(traceback.format_exc())
        
        with ai_tab4:
            st.subheader("📉 Predictive Analysis for Overall Process Performance")
            st.markdown("Analyze the overall impact of bug categories on process performance and identify critical activities.")
            
            if st.button("🔍 Analyze Overall Performance", type="primary"):
                with st.spinner("Analyzing overall process performance..."):
                    try:
                        from utils.category_prioritization import analyze_overall_process_performance
                        
                        performance_data = analyze_overall_process_performance(
                            df_filtered,
                            sla_threshold=sla_threshold
                        )
                        
                        if performance_data:
                            st.success("✅ Analysis completed!")
                            
                            # Overall KPIs
                            st.subheader("📊 Overall Process KPIs")
                            
                            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                            
                            with kpi_col1:
                                st.metric(
                                    "Average Resolution Time",
                                    f"{performance_data['overall_avg_duration']:.1f}h"
                                )
                            
                            with kpi_col2:
                                st.metric(
                                    "SLA Breach Rate",
                                    f"{performance_data['sla_breach_rate']:.1f}%",
                                    delta=f"-{performance_data['sla_breach_count']} cases"
                                )
                            
                            with kpi_col3:
                                st.metric(
                                    "Average Reassignments",
                                    f"{performance_data['avg_reassignments']:.1f}",
                                    help="Average number of different activities per case"
                                )
                            
                            with kpi_col4:
                                st.metric(
                                    "Rework Rate",
                                    f"{performance_data['rework_rate']:.1f}%",
                                    help="Percentage of cases that were reopened"
                                )
                            
                            # Category Impact Analysis
                            if not performance_data['category_impacts'].empty:
                                st.subheader("🎯 Category Impact on Overall Process")
                                
                                category_impacts = performance_data['category_impacts']
                                
                                # Impact visualization
                                import plotly.express as px
                                
                                fig_impact = px.bar(
                                    category_impacts,
                                    x='total_impact_hours',
                                    y='category',
                                    orientation='h',
                                    title='Total Impact by Category (Hours × Instances)',
                                    color='sla_breach_rate',
                                    color_continuous_scale='RdYlGn_r',
                                    labels={
                                        'total_impact_hours': 'Total Impact (hours)',
                                        'category': 'Category',
                                        'sla_breach_rate': 'SLA Breach Rate (%)'
                                    }
                                )
                                fig_impact.update_layout(height=400)
                                st.plotly_chart(fig_impact, use_container_width=True)
                                
                                # Category impact table
                                st.subheader("📋 Category Impact Details")
                                
                                display_impacts = category_impacts.copy()
                                display_impacts = display_impacts.round({
                                    'avg_duration': 1,
                                    'total_impact_hours': 1,
                                    'impact_percentage': 2,
                                    'sla_breach_rate': 1
                                })
                                
                                st.dataframe(
                                    display_impacts,
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            # Critical Activities
                            if not performance_data['critical_activities'].empty:
                                st.subheader("🔴 Most Critical Activities (Slowest)")
                                
                                critical_acts = performance_data['critical_activities']
                                
                                fig_critical = px.bar(
                                    critical_acts,
                                    x='avg_duration',
                                    y='activity',
                                    orientation='h',
                                    title='Top 10 Slowest Activities',
                                    color='frequency',
                                    color_continuous_scale='Reds',
                                    labels={
                                        'avg_duration': 'Average Duration (hours)',
                                        'activity': 'Activity',
                                        'frequency': 'Frequency'
                                    }
                                )
                                fig_critical.update_layout(height=400)
                                st.plotly_chart(fig_critical, use_container_width=True)
                                
                                st.dataframe(
                                    critical_acts.round({'avg_duration': 1}),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            
                            # Recommendations
                            st.subheader("💡 Recommendations")
                            
                            if performance_data['sla_breach_rate'] > 30:
                                st.error(f"⚠️ **High SLA Breach Rate ({performance_data['sla_breach_rate']:.1f}%)** - Urgent attention required!")
                                st.markdown("""
                                - Review and optimize slowest categories
                                - Allocate more resources to high-impact categories
                                - Consider process reengineering for critical activities
                                """)
                            
                            if performance_data['rework_rate'] > 15:
                                st.warning(f"⚠️ **High Rework Rate ({performance_data['rework_rate']:.1f}%)** - Quality issues detected")
                                st.markdown("""
                                - Investigate root causes of bug reopenings
                                - Improve testing and validation processes
                                - Consider code review improvements
                                """)
                            
                            if performance_data['avg_reassignments'] > 5:
                                st.info(f"ℹ️ **High Reassignment Rate ({performance_data['avg_reassignments']:.1f})** - Complex workflows")
                                st.markdown("""
                                - Consider workflow simplification
                                - Review activity dependencies
                                - Optimize handoff processes
                                """)
                            
                            if performance_data['sla_breach_rate'] < 10 and performance_data['rework_rate'] < 5:
                                st.success("✅ **Process is performing well!** Keep monitoring and maintain current practices.")
                                
                    except Exception as e:
                        st.error(f"Error analyzing overall performance: {str(e)}")
                        import traceback
                        with st.expander("Show detailed error"):
                            st.code(traceback.format_exc())
    
    with tab7:
        st.header("🎬 Process Animation - Token Replay")
        st.markdown("Visualize how cases flow through the process over time, just like in Fluxicon Disco!")
        
        from utils.animation import create_process_animation, create_token_replay
        from utils.advanced_animation import create_disco_style_animation
        
        # Animation settings
        st.subheader("⚙️ Animation Settings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            animation_type = st.selectbox(
                "Animation Type",
                options=['Disco-Style (Recommended)', 'Time-based Flow', 'Token Replay'],
                help="Disco-Style: Tokens move on arcs at variable speed based on duration. Time-based: Shows cases appearing over time. Token Replay: Step-by-step event replay"
            )
        
        with col2:
            max_cases = st.slider(
                "Number of Cases",
                min_value=10,
                max_value=100,
                value=30,
                help="More cases = slower animation"
            )
        
        with col3:
            animation_speed = st.slider(
                "Animation Speed (ms)",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Lower = faster"
            )
        
        st.divider()
        
        # Generate animation button
        if st.button("🎬 Generate Animation", type="primary"):
            with st.spinner("Creating animation... This may take a few seconds..."):
                try:
                    if animation_type == 'Disco-Style (Recommended)':
                        fig_anim = create_disco_style_animation(
                            df_filtered,
                            max_cases=max_cases,
                            animation_speed=animation_speed,
                            sla_threshold=sla_threshold
                        )
                    elif animation_type == 'Time-based Flow':
                        fig_anim = create_process_animation(
                            df_filtered,
                            max_cases=max_cases,
                            animation_speed=animation_speed
                        )
                    else:
                        fig_anim = create_token_replay(
                            df_filtered,
                            selected_case_id=None,
                            animation_speed=animation_speed
                        )
                    
                    st.plotly_chart(fig_anim, use_container_width=True)
                    
                    st.success("✅ Animation created! Use the play button to start.")
                    
                    # Instructions
                    with st.expander("📖 How to use the animation"):
                        if animation_type == 'Disco-Style (Recommended)':
                            st.markdown("""
                            **Controls:**
                            - **▶️ Play**: Start the animation
                            - **⏸️ Pause**: Pause at current frame
                            - **⏮️ Reset**: Return to beginning
                            - **Slider**: Manually navigate through frames
                            
                            **What you're seeing (Disco-Style):**
                            - **Blue circles**: Activities in the process
                            - **Colored arrows**: Transitions (Green=fast, Orange=medium, Red=slow)
                            - **Colored dots**: Cases (bugs) moving through the process
                            - **Token movement**:
                              - ⚡ **Fast tokens** on green arrows: Short transitions (performing well)
                              - 🐌 **Slow tokens** on red arrows: Long transitions (bottlenecks)
                              - The speed at which a token moves on an arrow reflects the real duration!
                            - **Dot colors**:
                              - 🔴 Red: Critical priority
                              - 🟠 Orange: High priority
                              - 🟡 Yellow: Medium priority
                              - 🟢 Green: Low priority
                            
                            **What makes this special:**
                            - Tokens move ALONG the arrows (not just jumping between nodes)
                            - Speed is proportional to transition duration
                            - Visual identification of slow vs fast paths
                            - Just like Fluxicon Disco!
                            
                            **Tips:**
                            - Watch tokens moving slowly on red arrows = bottlenecks
                            - Watch tokens moving quickly on green arrows = efficient
                            - Hover over dots to see case details and progress
                            - Adjust speed for better visualization
                            - Use 20-30 cases for optimal viewing
                            """)
                        else:
                            st.markdown("""
                            **Controls:**
                            - **▶️ Play**: Start the animation
                            - **⏸️ Pause**: Pause at current frame
                            - **⏮️ Reset**: Return to beginning
                            - **Slider**: Manually navigate through frames
                            
                            **What you're seeing:**
                            - **Blue circles**: Activities in the process
                            - **Colored dots**: Cases (bugs) moving through the process
                            - **Dot colors**:
                              - 🔴 Red: Critical priority
                              - 🟠 Orange: High priority
                              - 🟡 Yellow: Medium priority
                              - 🟢 Green: Low priority
                            
                            **Tips:**
                            - Hover over dots to see case details
                            - Watch for bottlenecks where dots accumulate
                            - Adjust speed for better visualization
                            - Use fewer cases for smoother animation
                            """)
                    
                except Exception as e:
                    st.error(f"Error creating animation: {str(e)}")
                    import traceback
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())
        
        else:
            st.info("👆 Click 'Generate Animation' to create the process animation")
            
            # Show preview
            st.markdown("### 🎥 Preview")
            st.markdown("""
            The animation will show:
            - **Cases flowing** through your bug resolution process
            - **Real-time visualization** of how bugs move between activities
            - **Color-coded priorities** to identify critical issues
            - **Interactive controls** to play, pause, and navigate
            
            **🆕 Disco-Style Animation (Recommended):**
            - ⚡ **Variable token speed**: Tokens move FAST on short transitions, SLOW on long transitions
            - 🎯 **Visual bottleneck identification**: See where tokens slow down
            - 🗺️ **Hierarchical layout**: Top-to-bottom flow like Disco
            - 🌈 **Color-coded arrows**: Green (fast), Orange (medium), Red (slow/bottleneck)
            - 📍 **Position on arc**: Shows exact progress through each transition
            
            This is the **token replay** feature from Fluxicon Disco, enhanced with:
            - Real-time speed visualization
            - Arc-based movement (not just node-to-node)
            - Performance-based coloring
            - Hierarchical process layout
            """)
            
            # Example image placeholder
            st.image("https://via.placeholder.com/800x400/3498db/ffffff?text=Process+Animation+Preview", 
                    caption="Animation will display here", use_column_width=True)
    
    # Footer with export options
    st.divider()
    st.header("📥 Export & Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export bottlenecks
        try:
            bottlenecks_csv = identify_bottlenecks(df_filtered, sla_threshold, top_n=20).to_csv(index=False)
            st.download_button(
                label="Download Bottlenecks Report",
                data=bottlenecks_csv,
                file_name=f"bottlenecks_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except:
            pass
    
    with col2:
        # Export category stats
        try:
            category_csv = calculate_category_statistics(df_filtered).to_csv(index=False)
            st.download_button(
                label="Download Category Statistics",
                data=category_csv,
                file_name=f"category_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except:
            pass
    
    with col3:
        # Export filtered data
        filtered_csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=filtered_csv,
            file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Sidebar footer
with st.sidebar:
    st.divider()
    st.markdown("""
    ### 📖 About
    **Process Mining Dashboard v1.0**
    
    Built with:
    - Streamlit
    - pm4py
    - Plotly
    - Pandas
    
    💡 **Tips**: 
    - Use filters to focus on specific categories
    - Adjust SLA threshold to match your requirements
    - Export reports for offline analysis
    """)

