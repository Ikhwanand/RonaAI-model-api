FROM python:3.12-slim 

WORKDIR /

# Copy requirements and install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt 

# Copy application code
COPY . . 


# Expose the port the app runs on 
EXPOSE 8099

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8099"]
