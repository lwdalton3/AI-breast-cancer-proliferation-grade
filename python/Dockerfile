FROM python:3.9

# Install packages required for the project
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Set the working directory to bind-mount where project files are
WORKDIR /backend

# Starts Flask backend in dev mode
ENTRYPOINT python run.py
