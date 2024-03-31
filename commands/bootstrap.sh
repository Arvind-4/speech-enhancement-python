#!/bin/bash

# Updated Packages
sudo apt update -y
sudo apt upgrade -y

# Add PPA for Python 3.7
sudo add-apt-repository ppa:deadsnakes/ppa -y

# Update Packages
sudo apt update -y

# Install Python 3.7
sudo apt install python3.7 -y

# Install Python 3.7 Development Tools
sudo apt install python3.7-dev python3.7-venv python3.7-distutils python3.7-lib2to3 -y

python3.7 -m pip install virtualenv