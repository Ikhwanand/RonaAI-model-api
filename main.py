from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
import uvicorn
from load_model import validate_model

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)

@app.on_event("startup")
async def startup_event():
    validate_model()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8099)
