#!/bin/bash

# Pull the latest changes from the repository
git pull

# Check if the virtual environment exists
if [ ! -d "env" ]; then
  # Create the virtual environment
  python3 -m venv env
fi

# Activate the virtual environment
source env/bin/activate

# Install the required packages
pip install -r requirements.txt

# Run the application
python app.py