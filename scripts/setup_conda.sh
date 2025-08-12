#!/bin/bash
# Setup script for Conda environment and initial configuration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DNA Project Conda Setup ===${NC}"

# Step 1: Check for Conda and install if needed
echo -e "\n${YELLOW}Step 1: Checking for Conda...${NC}"

if command -v conda >/dev/null 2>&1; then
    echo "Conda found at: $(which conda)"
    eval "$(conda shell.bash hook)"
elif [ -x "$HOME/miniconda3/bin/conda" ]; then
    echo "Conda found at: $HOME/miniconda3/bin/conda"
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
else
    echo -e "${YELLOW}Conda not found. Installing Miniconda...${NC}"
    cd "$HOME"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm -f miniconda.sh
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda config --set auto_activate_base false
    echo -e "${GREEN}Miniconda installed successfully!${NC}"
fi

# Step 2: Create/Update environment
echo -e "\n${YELLOW}Step 2: Setting up dnaenv environment...${NC}"

ENV_NAME=dnaenv
ENV_FILE="/home/mch/dna/environment.yml"

if [ -f "$ENV_FILE" ]; then
    echo "Using environment.yml file..."
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo "Environment exists, updating..."
        conda env update -f "$ENV_FILE"
    else
        echo "Creating new environment..."
        conda env create -f "$ENV_FILE"
    fi
else
    echo "Creating environment without yml file..."
    conda create -y -n "$ENV_NAME" python=3.11 pandas pyarrow duckdb jupyterlab ipykernel tqdm nbformat
fi

# Step 3: Activate and configure
echo -e "\n${YELLOW}Step 3: Configuring environment...${NC}"

conda activate "$ENV_NAME"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# Step 4: Show environment info
echo -e "\n${GREEN}Environment setup complete!${NC}"
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"

# Step 5: Provide next steps
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Run data conversion:"
echo "   python /home/mch/dna/scripts/convert_to_parquet.py"
echo ""
echo "2. Start Jupyter Lab:"
echo "   jupyter lab --notebook-dir=/home/mch/dna/notebooks --ServerApp.root_dir=/home/mch/dna"
echo ""
echo "3. Or run quick analysis:"
echo "   python -c \"import duckdb; con=duckdb.connect('/home/mch/dna/artifacts/dna.duckdb'); con.sql('SELECT * FROM barcode_stats').show()\""