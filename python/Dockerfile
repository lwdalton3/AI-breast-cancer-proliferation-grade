FROM python:3.8

# Install packages required for the project
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /
RUN pip install  -r /requirements.txt

# Used for development on my M1 mac
RUN pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html

# Set the working directory to bind-mount where project files are
WORKDIR /backend

# Starts Flask backend in dev mode
ENTRYPOINT python run.py
