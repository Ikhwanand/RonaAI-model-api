from fastapi import APIRouter, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse, FileResponse
from load_model import get_model_cache, visualize_prediction
import tempfile
import os
from pathlib import Path

router = APIRouter()

@router.post("/classify-face")
async def classify_face(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Get the model
        model = get_model_cache().model

        # Perform prediction
        prediction_result = model.predict(temp_file_path)


        # Prepare the response
        response = {
            "prediction": prediction_result,
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

