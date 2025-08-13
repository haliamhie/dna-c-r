#!/bin/bash

# CRISPR AI Platform Startup Script
# Starts both backend and frontend servers

echo "Starting CRISPR AI Platform..."
echo "==============================="

# Setup conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dnaenv

# Kill any existing servers
echo "Stopping any existing servers..."
pkill -f "python app.py" 2>/dev/null
pkill -f "http.server 8080" 2>/dev/null
sleep 2

# Start backend
echo "Starting backend API server on port 8000..."
cd /home/mch/dna/backend
python app.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend server on port 8080..."
cd /home/mch/dna/frontend
python3 -m http.server 8080 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for servers to start
sleep 2

# Check status
echo ""
echo "Checking server status..."
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✓ Backend API is running at http://localhost:8000"
else
    echo "✗ Backend API failed to start"
fi

if curl -s http://localhost:8080/ > /dev/null; then
    echo "✓ Frontend is running at http://localhost:8080"
else
    echo "✗ Frontend failed to start"
fi

echo ""
echo "==============================="
echo "CRISPR AI Platform is ready!"
echo ""
echo "Access points:"
echo "  - Frontend UI: http://localhost:8080"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "To stop servers, run: pkill -f 'python app.py' && pkill -f 'http.server 8080'"
echo ""