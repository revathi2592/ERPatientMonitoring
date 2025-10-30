# Use the official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy files
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 8080
EXPOSE 8080

# Run the app with Gunicorn
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 agent1:app
