# Rubric-based scoring model - Flask app for Dockploy
FROM python:3.11-slim

# Prevent Python from writing pyc and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Listen on port 80 and bind to all interfaces (for Docker)
ENV PORT=80
ENV HOST=0.0.0.0
ENV FLASK_DEBUG=false

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (include final_model.pkl in project root for predictions)
COPY . .

# Expose port 80 for Dockploy
EXPOSE 80

# Run the app (uses PORT and HOST from env)
CMD ["python", "app.py"]
