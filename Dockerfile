# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for pip and git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run the main app (change to app.py or public_chatbot.py depending on your entry point)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "public_chatbot:app"]
