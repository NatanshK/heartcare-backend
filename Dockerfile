# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose Port
EXPOSE 8080

# Define the Command to Run the Application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT:-8080}", "--workers", "1", "--threads", "8", "--timeout", "0"]