#!/bin/bash
# Shell script for Linux/Mac users to run the complete pipeline

echo "========================================"
echo "Galaxy Morphology Unlearning Pipeline"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from python.org"
    exit 1
fi

echo "[1/4] Testing setup..."
python3 test_setup.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Setup test failed"
    echo "Please fix the issues above"
    exit 1
fi

echo ""
echo "[2/4] Running main pipeline..."
echo "This will take 20-30 minutes. Please wait..."
echo ""
python3 main.py
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Pipeline failed"
    exit 1
fi

echo ""
echo "[3/4] Pipeline complete!"
echo ""
echo "Generated files:"
ls -1 *.pth 2>/dev/null || echo "  (models will appear here)"
ls -1 *.csv 2>/dev/null || echo "  (logs will appear here)"
echo ""

echo "[4/4] Would you like to launch the web app? (y/n)"
read -p "Enter choice: " launch

if [ "$launch" = "y" ] || [ "$launch" = "Y" ]; then
    echo ""
    echo "Launching Streamlit app..."
    echo "Open your browser to http://localhost:8501"
    echo "Press Ctrl+C to stop the server"
    echo ""
    streamlit run app.py
else
    echo ""
    echo "To launch the web app later, run:"
    echo "  streamlit run app.py"
fi

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
