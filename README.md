# 🔍 Process Mining & AI Dashboard

An intelligent dashboard for analyzing bug workflows using Process Mining and AI techniques. Built with Streamlit and pm4py.

![Dashboard Preview](https://img.shields.io/badge/Status-Production_Ready-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Features

### Core Functionality
- **📊 KPI Dashboard**: Real-time metrics including resolution time, SLA compliance, and reopening rates
- **🗺️ Process Map**: Interactive workflow visualization with color-coded bottleneck indicators
- **🔥 Heatmap Analysis**: Identify slow activities by category and priority
- **⚠️ Bottleneck Detection**: Automatically identify process inefficiencies
- **🔄 Variant Analysis**: Discover the most common paths through your workflow
- **📈 Statistical Distributions**: Comprehensive data visualization and analysis
- **📅 Temporal Analysis**: Track trends and patterns over time
- **🔄 Loop Detection**: Identify rework and repeated activities

### Advanced Features
- Multi-dimensional filtering (category, priority, severity, date range)
- Customizable SLA thresholds
- Export capabilities (CSV reports)
- Interactive Plotly visualizations
- Real-time data validation

### 🎬 Process Animation (NEW - Disco-Style!)
- **Token Replay**: Visualize cases flowing through the process in real-time
- **Two Animation Modes**: Time-based flow and step-by-step event replay
- **Interactive Controls**: Play, pause, reset, and navigate frame-by-frame
- **Color-Coded Tokens**: Priority-based coloring (Critical, High, Medium, Low)
- **Bottleneck Visualization**: See where cases accumulate
- **Adjustable Speed**: Control animation speed for detailed analysis
- **Hover Details**: View case information by hovering over tokens

### 🤖 AI & Predictive Analysis
- **Machine Learning Models**: Train Random Forest, Gradient Boosting, or Linear Regression models
- **Duration Prediction**: Predict resolution time for new bugs before assignment
- **Risk Assessment**: Calculate complexity score and risk level (Low/Medium/High/Critical)
- **Smart Recommendations**: Get actionable insights based on predictions
- **Model Comparison**: Evaluate different algorithms side-by-side
- **Feature Importance**: Understand which factors influence bug duration
- **Historical Analysis**: Leverage past data for better predictions

## 📋 Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## 🚀 Installation

### Option 1: Docker (Recommandée - La Plus Simple) 🐳

**Windows:**
```cmd
docker-run.bat
```

**Linux/Mac:**
```bash
chmod +x docker-run.sh
./docker-run.sh
```

**Ou avec Docker Compose:**
```bash
docker-compose up -d
```

➡️ **Voir [README_DOCKER.md](README_DOCKER.md) pour plus de détails**

---

### Option 2: Installation Python Locale

#### 1. Clone or download this repository

```bash
git clone <repository-url>
cd DISCO
```

#### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Run the application

```bash
streamlit run app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
DISCO/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── sample_bug_events.csv       # Sample dataset
│
└── utils/                      # Utility modules
    ├── __init__.py
    ├── data_loader.py          # CSV loading and validation
    ├── metrics.py              # KPI calculations
    ├── process_mining.py       # pm4py integration and DFG
    └── visualizations.py       # Plotly visualizations
```

## 📊 Data Format

Your CSV file should contain the following columns:

| Column | Alternative Names | Type | Description | Example |
|--------|-------------------|------|-------------|---------|
| `case_id` | `case:concept:name` | String | Unique bug identifier | BUG-001, ISSUE-123 |
| `activity` | `concept:name` | String | Process step/activity | Open, Analyze, Fix, Test, Close |
| `timestamp` | `time:timestamp` | DateTime | Event timestamp | 2024-01-15 14:30:00 |
| `category` | `Category` | String | Bug category | Backend, Frontend, Database, API |
| `priority` | `Priority` | String | Bug priority | Critical, High, Medium, Low |
| `severity` | `Severity` | String | Bug severity | Blocker, Major, Minor, Trivial |

### Optional Columns
- `assignee`: Person assigned to the bug
- `team`: Team handling the bug
- `resolution_time`: Actual resolution time
- Any other metadata fields

### Supported Formats

- **CSV** (.csv) - Comma-separated values
- **Excel** (.xlsx, .xls) - Microsoft Excel format

### Example File

```csv
case_id,activity,timestamp,category,priority,severity
BUG-001,Open,2024-01-15 09:00:00,Backend,High,Major
BUG-001,Analyze,2024-01-15 10:30:00,Backend,High,Major
BUG-001,Fix,2024-01-15 14:00:00,Backend,High,Major
BUG-001,Test,2024-01-16 09:00:00,Backend,High,Major
BUG-001,Close,2024-01-16 11:00:00,Backend,High,Major
BUG-002,Open,2024-01-15 10:00:00,Frontend,Medium,Minor
```

## 🎮 Usage Guide

### 1. Upload Data
- Click "Browse files" in the sidebar
- Select your CSV event log file
- The system will automatically validate the data

### 2. Apply Filters
- **Category**: Filter by bug categories (Backend, Frontend, etc.)
- **Priority**: Filter by priority levels
- **Severity**: Filter by severity levels
- **Date Range**: Select specific time period
- **SLA Threshold**: Adjust the acceptable resolution time (in hours)

### 3. Explore Visualizations

#### Process Map Tab
- View the workflow as a directed graph
- **Green edges**: Activities within SLA
- **Orange edges**: Activities approaching SLA limit (50-100%)
- **Red edges**: Activities exceeding SLA threshold
- Edge thickness represents frequency

#### Heatmap & Bottlenecks Tab
- Heatmap shows average duration by activity and category
- Bottlenecks table lists the slowest transitions
- Color coding: Red (slow) to Green (fast)

#### Distributions Tab
- Case duration histogram
- Priority distribution pie chart
- Activity frequency bar chart
- Category performance comparison

#### Variants & Loops Tab
- Most common process paths
- Rework detection (activities that repeat)
- Process variant statistics

#### Temporal Analysis Tab
- Timeline of critical cases (exceeding SLA)
- Case volume trends over time

#### Animation Tab 🎬 (NEW!)
- **Token Replay**: Disco-style process animation
- **Time-based Flow**: See cases appearing and moving chronologically
- **Interactive Controls**: Play, pause, reset, frame-by-frame navigation
- **Bottleneck Visualization**: Watch where cases accumulate
- **Priority Color-Coding**: Red (Critical), Orange (High), Yellow (Medium), Green (Low)
- **Adjustable Settings**: Number of cases (10-100), animation speed (50-1000ms)
- **Two Modes**:
  - Time-based Flow: Shows process evolution over time
  - Token Replay: Step-by-step event replay

**Quick Start:**
1. Go to "🎬 Animation" tab
2. Select animation type and settings
3. Click "Generate Animation"
4. Press ▶️ Play and watch your process come to life!

**Documentation**: See [ANIMATION_QUICK_START.md](ANIMATION_QUICK_START.md) for 5-minute tutorial

### 4. Export Reports
- Download bottlenecks analysis as CSV
- Export category statistics
- Download filtered dataset

## 🎨 Customization

### Adjusting SLA Thresholds
Use the slider in the sidebar to set your SLA threshold (1-168 hours). The process map and bottleneck detection will automatically update.

### Color Coding
- 🟢 **Green**: Within 50% of SLA (performing well)
- 🟠 **Orange**: 50-100% of SLA (needs attention)
- 🔴 **Red**: Exceeds SLA (critical bottleneck)

## 🧪 Sample Data

The application includes a sample data generator:
1. Open the dashboard without uploading a file
2. Click "Generate Sample CSV" button
3. Download the generated sample file
4. Upload it to explore the features

Alternatively, use the included `sample_bug_events.csv` file.

## 🔧 Troubleshooting

### Import Errors
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt --upgrade
```

### Graphviz Issues
If you encounter graphviz errors:

**Windows:**
```bash
# Download and install Graphviz from:
# https://graphviz.org/download/
# Add to PATH: C:\Program Files\Graphviz\bin
```

**Linux:**
```bash
sudo apt-get install graphviz
```

**Mac:**
```bash
brew install graphviz
```

### Large Files
For datasets with 100k+ rows:
- The application uses caching for performance
- First load may take longer
- Subsequent analyses will be faster

## 📈 Performance Tips

1. **Filter Data**: Use filters to focus on specific subsets
2. **Date Ranges**: Analyze shorter time periods for faster processing
3. **Caching**: Streamlit automatically caches calculations
4. **Browser**: Use Chrome or Firefox for best performance

## 🛠️ Technology Stack

- **[Streamlit](https://streamlit.io/)**: Web application framework
- **[pm4py](https://pm4py.fit.fraunhofer.de/)**: Process mining library
- **[Plotly](https://plotly.com/)**: Interactive visualizations
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation
- **[NetworkX](https://networkx.org/)**: Graph analysis
- **[NumPy](https://numpy.org/)**: Numerical computing

## 📚 Process Mining Concepts

### What is Process Mining?
Process mining is a technique that analyzes event logs to discover, monitor, and improve business processes.

### Key Metrics
- **Case Duration**: Time from start to end of a case
- **Activity Duration**: Time spent in each activity
- **Bottleneck**: Activities or transitions that slow down the process
- **Variant**: A unique path through the process
- **SLA Compliance**: Percentage of cases meeting the threshold

### DFG (Directly-Follows Graph)
A process map showing activities and their direct connections, annotated with frequency and performance metrics.

## 🔮 Future Enhancements

- [ ] Machine Learning predictions (Random Forest, XGBoost)
- [ ] Anomaly detection
- [ ] Real-time data streaming
- [ ] Advanced conformance checking
- [ ] Resource utilization analysis
- [ ] PDF report generation
- [ ] Multi-language support
- [ ] Dark mode theme

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created for bug workflow analysis and process optimization.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the sample data format
3. Open an issue on GitHub

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Process mining powered by [pm4py](https://pm4py.fit.fraunhofer.de/)
- Visualizations created with [Plotly](https://plotly.com/)

---

**Happy Process Mining! 🚀**

