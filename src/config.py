from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  thermal_capacity_level: str = "medium"


settings = Settings()
