FROM python:3.11

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model during build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY . .

# Set writable cache paths (to avoid /.cache permission errors)
ENV TRANSFORMERS_CACHE=/tmp/transformers
ENV HF_HOME=/tmp/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/st_cache

# Expose port expected by Hugging Face
EXPOSE 7860

# Run your Flask app
CMD ["python", "app.py"]
