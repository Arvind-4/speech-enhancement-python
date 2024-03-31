import pathlib
from functools import lru_cache
from pydantic import BaseSettings


class Settings(BaseSettings):
    BASE_DIR: pathlib.Path = pathlib.Path(__file__).resolve(strict=True).parent.parent
    MODEL_FILE_UNET: pathlib.Path = BASE_DIR / "weights" / "model_unet.h5"
    MODEL_JSON_UNET: pathlib.Path = BASE_DIR / "weights" / "model_unet.json"
    MODEL_FILE_GAN: pathlib.Path = BASE_DIR / "weights" / "model_gan.h5"
    MODEL_JSON_GAN: pathlib.Path = BASE_DIR / "weights" / "model_gan.json"
    AUDIO_DIR_PREDICTION: pathlib.Path = BASE_DIR / "test_data" / "test"
    DIR_SAVE_PREDICTION_UNET: pathlib.Path = BASE_DIR / "test_data" / "saved_predictions" / "unet"
    DIR_SAVE_PREDICTION_GAN: pathlib.Path = BASE_DIR / "test_data" / "saved_predictions" / "gan"
    SAMPLE_RATE: int = 8000
    MIN_DURATION: float = 1.0
    FRAME_LENGTH: int = 8064
    HOP_LENGTH_FRAME: int = 8064
    N_FFT: int = 255
    HOP_LENGTH_FFT: int = 63


@lru_cache()
def get_settings():
    settings = Settings()
    return settings
