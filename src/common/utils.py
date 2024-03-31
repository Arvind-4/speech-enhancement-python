from config.config import get_settings

settings = get_settings()


def write_file(file) -> bool:
    try:
        with open(settings.AUDIO_DIR_PREDICTION + "/" + file.filename, "wb") as f:
            f.write(file.file.read())
        return True
    except Exception as e:
        print(e)
        return False
