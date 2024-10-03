# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install ffmpeg for saving animations
RUN apt-get update && apt-get install -y ffmpeg

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Command to run your Python script
CMD ["python", "testingStandAlone.py"]
