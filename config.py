from pydantic_settings import BaseSettings
import os 



class Settings(BaseSettings):
    PROJECT_NAME: str = "RonaAI-Model-API"
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: list = ["*"]
    
    class Config:
        case_sensitive = True 
        
        
settings = Settings()