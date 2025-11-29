#!/bin/bash
# AgriClimate Insight Portal - Startup Script

echo "🌾 AgriClimate Insight Portal"
echo "=============================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "⚠️ Python version $python_version detected. Recommended: Python 3.11+"
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Run tests
echo "🧪 Running tests..."
python test_app.py

# Start the application
echo ""
echo "🚀 Starting AgriClimate Insight Portal..."
echo "   The application will be available at: http://localhost:8501"
echo "   Press Ctrl+C to stop the application"
echo ""

streamlit run app.py
