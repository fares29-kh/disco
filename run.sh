#!/bin/bash
# Script de lancement pour Linux/Mac
# Process Mining Dashboard

echo "========================================"
echo " Process Mining Dashboard Launcher"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    echo "[OK] Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate
echo ""

# Install/Update dependencies
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "[OK] Dependencies installed!"
echo ""

# Launch Streamlit app
echo "[INFO] Starting Process Mining Dashboard..."
echo "[INFO] Opening http://localhost:8501 in your browser..."
echo ""
echo "Press CTRL+C to stop the application"
echo "========================================"
echo ""

streamlit run app.py

