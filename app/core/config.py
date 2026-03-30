import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    DATABASE_URL: str = "sqlite:///./sql_app.db"
    MODEL_NAME: str = "intfloat/multilingual-e5-base"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
