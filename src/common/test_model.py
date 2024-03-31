import os
import json
import tensorflow as tf
import soundfile as sf
from data_tools import scaled_in, inv_scaled_ou
from data_tools import (
    audio_files_to_numpy,
    numpy_audio_to_matrix_spectrogram,
    matrix_spectrogram_to_numpy_audio,
)
from config.config import get_settings

settings = get_settings()

audio_dir_prediction = str(settings.AUDIO_DIR_PREDICTION)
sample_rate = settings.SAMPLE_RATE
min_duration = settings.MIN_DURATION
frame_length = settings.FRAME_LENGTH
hop_length_frame = settings.HOP_LENGTH_FRAME
n_fft = settings.N_FFT
hop_length_fft = settings.HOP_LENGTH_FFT


def prediction(model_file: str = None, json_file: str = None, dir_save_prediction: str = None):
    with open(json_file) as f:
        json_config = f.read()
        loaded_model = tf.keras.models.model_from_json(json_config)

    loaded_model.load_weights(model_file)
    # with open(json_file, "r") as json_file:
    #     model_config = json.load(json_file)
    # model = tf.keras.models.model_from_json(json.dumps(model_config))
    # print("MODEL", model)
    # loaded_model = model.load_weights(model_file)
    # print("LOADED MODEL", loaded_model)
    print("Model loaded from disk", loaded_model)
    if not os.path.exists(dir_save_prediction):
        os.makedirs(dir_save_prediction)

    for file in os.listdir(audio_dir_prediction):
        print("Denoising audio %s" % file)
        audio = audio_files_to_numpy(
            audio_dir_prediction,
            [file],
            sample_rate,
            frame_length,
            hop_length_frame,
            min_duration,
        )

        dim_square_spec = int(n_fft / 2) + 1
        m_amp_db_audio, m_pha_audio = numpy_audio_to_matrix_spectrogram(
            audio, dim_square_spec, n_fft, hop_length_fft
        )
        X_in = scaled_in(m_amp_db_audio)
        X_in = X_in.reshape(X_in.shape[0], X_in.shape[1], X_in.shape[2], 1)
        X_pred = loaded_model.predict(X_in)
        inv_sca_X_pred = inv_scaled_ou(X_pred)
        X_denoise = m_amp_db_audio - inv_sca_X_pred[:, :, :, 0]
        audio_denoise_recons = matrix_spectrogram_to_numpy_audio(
            X_denoise, m_pha_audio, frame_length, hop_length_fft
        )
        nb_samples = audio_denoise_recons.shape[0]
        denoise_long = audio_denoise_recons.reshape(1, nb_samples * frame_length) * 10
        sf.write(dir_save_prediction + "/" + file, denoise_long[0, :], sample_rate)

