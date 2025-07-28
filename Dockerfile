FROM --platform=linux/amd64 python:3.10

# Set working directory inside container
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy language model
RUN python -m spacy download en_core_web_sm

# Copy ALL remaining project files and directories
COPY . .

# Download the model during the build process
# RUN python download_model.py

# Set default command to run your main script
CMD ["python", "main.py"]