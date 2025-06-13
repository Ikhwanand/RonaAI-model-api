# RonaAI Model API

## Project Description

RonaAI Model API is a FastAPI-based application that provides an API for facial image classification. This project allows users to upload facial images and receive classification results along with a visualized output image.

## Features

- Face classification API endpoint
- Image visualization of classification results
- JSON response with prediction results and image URL

## Installation

1. Clone this repository:

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv source venv/bin/activate # On Windows use venv\Scripts\activate
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```


## Usage

1. Start the FastAPI server:

```bash
python main.py
```


2. The API will be available at `http://localhost:8099`

3. Use the `/classify-face` endpoint to upload and classify facial images:
- Method: POST
- URL: `http://localhost:8000/classify-face`
- Body: form-data with key 'file' and value as the image file

4. The API will return a JSON response with:
- Prediction results
- URL to view the classified image

## API Endpoints

- `POST /classify-face`: Upload and classify a facial image
- `GET /classified-image`: Retrieve the classified image

## Project Structure

- `main.py`: Main application file
- `routes.py`: API route definitions
- `config.py`: Configuration settings
- `load_model.py`: Model loading and prediction functions
- `requirements.txt`: List of project dependencies

## Dependencies

- Python 3.12+
- FastAPI
- Uvicorn
- (Other dependencies as listed in requirements.txt)

## Contributing

Contributions to the RonaAI Model API project are welcome. Please feel free to submit a Pull Request.

