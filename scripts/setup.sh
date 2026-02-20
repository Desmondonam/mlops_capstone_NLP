#!/bin/bash

# ─── MLOps Capstone Setup Script ─────────────────────────────────────────
set -e

echo "╔════════════════════════════════════════╗"
echo "║   MLOps Capstone Setup Script           ║"
echo "╚════════════════════════════════════════╝"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✅ $1 is installed${NC}"
    else
        echo -e "${RED}❌ $1 is NOT installed - please install it${NC}"
        exit 1
    fi
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."
check_command docker
check_command python3

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "${GREEN}✅ Python $PYTHON_VERSION${NC}"

# Create virtual environment
echo ""
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download NLTK data
echo ""
echo "Downloading NLTK data..."
python3 -c "
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
print('✅ NLTK data downloaded')
"

# Create directories
echo ""
echo "Creating project directories..."
mkdir -p data/{raw,processed,features}
mkdir -p artifacts/{models,checkpoints}
mkdir -p mlruns
mkdir -p logs
mkdir -p configs/grafana/{provisioning,dashboards}

# Initialize DVC (if installed)
if command -v dvc &> /dev/null; then
    echo "Initializing DVC..."
    dvc init --no-scm 2>/dev/null || true
    echo -e "${GREEN}✅ DVC initialized${NC}"
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup Complete! 🎉                    ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Next steps:                            ║${NC}"
echo -e "${GREEN}║  1. source venv/bin/activate            ║${NC}"
echo -e "${GREEN}║  2. docker-compose up -d                ║${NC}"
echo -e "${GREEN}║  3. python -m src.training.trainer      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
