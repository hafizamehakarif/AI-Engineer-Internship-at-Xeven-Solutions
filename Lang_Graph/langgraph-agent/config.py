from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Runtime settings for the LangGraph agent application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str
    database_url: str
    available_model: str = "openai/gpt-oss-120b"  #  dynamic field added
    app_host: str = "0.0.0.0"
    app_port: int = 8000

def get_settings() -> Settings:
    """Return application settings instance."""
    return Settings()