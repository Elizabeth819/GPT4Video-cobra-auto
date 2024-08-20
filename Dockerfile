# Use an official Python runtime as a parent image  
FROM python:3.10-slim
  
# Set the working directory in the container  
WORKDIR /usr/src/app  
  
# Install necessary system packages  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    ffmpeg \  
    libsm6 \  
    libxext6 \  
    make \
    cmake \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*  
  
# Copy the current directory contents into the container at /usr/src/app  
COPY ./ActionSummary.py ./ActionSummary.py
COPY ./ChapterAnalyzer.py ./ChapterAnalyzer.py
COPY ./VO.py ./VO.py
COPY ./requirements.txt ./requirements.txt

  
# Install any needed packages specified in requirements.txt  
# Note: You should create a requirements.txt file with the necessary Python packages.  
# For this code, we would have packages like opencv-python-headless, moviepy, openai, python-dotenv, requests, etc.  
RUN pip install --no-cache-dir -r requirements.txt  
RUN pip install face-recognition
  
# Set environment variables from the provided 'env' section  

  
# Run the application  
ENTRYPOINT ["/bin/bash"]  
