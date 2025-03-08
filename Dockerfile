# Use official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy Python script and requirements file
COPY tokenization.py /app/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir nltk
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Python script
CMD ["python", "tokenization.py"]
