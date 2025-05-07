#!/usr/bin/env bash

# Create project folder
mkdir thermos-prototype
cd thermos-prototype

# Set up a Python virtual environment
python3 -m venv venv

# Activate the virtual environment (for Windows)
.\venv\Scripts\activate


# Install required Python packages
pip install numpy pandas matplotlib tensorflow scikit-learn psutil
