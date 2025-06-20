#!/bin/bash

# Update pip and install dependencies
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Download the English language model for spaCy
python3 -m spacy download en_core_web_sm