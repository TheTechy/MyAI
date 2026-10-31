#!/bin/bash
clear
# Navigate to the specified development directory
echo "Current directory: $(pwd)"

# Confirm before proceeding
read -p "Continue here? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Exiting."
    exit 1
fi

# Clean up compiled Python files (.pyc)
echo "Removing .pyc files..."
find . -name "*.pyc" -delete

# Clean up __pycache__ directories
echo "Removing __pycache__ directories..."
find . -name "__pycache__" -type d -exec rm -rf {} +

# Run the main application script
echo "Starting application..."
sleep 3
.venv/bin/python app.py