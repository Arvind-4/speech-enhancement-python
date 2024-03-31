import numpy as np
from matplotlib import pyplot as plt


class SignalAnalysis:
    time_unit: str = "second (ms)"
    amplitude_unit: str = "hertz (Hz)"

    def __init__(self, clean_signal, noise_signal) -> None:
        self.clean_signal = clean_signal
        self.noise_signal = noise_signal

    def signal_to_noise_ratio(self):
        snr = 20 * np.log10(
            np.mean(np.square(self.clean_signal))
            / np.mean(np.square(self.noise_signal))
        )
        return abs(self.round_to_three_decimal_points(snr))

    def percent_noise_removed(self):
        if self.noise_signal.shape[0] != self.clean_signal.shape[0]:
            if self.noise_signal.shape[0] < self.clean_signal.shape[0]:
                padding = np.zeros(
                    self.clean_signal.shape[0] - self.noise_signal.shape[0]
                )
                self.noise_signal = np.concatenate((self.noise_signal, padding))
            else:
                self.noise_signal = self.noise_signal[: self.clean_signal.shape[0]]

        noise = self.clean_signal - self.noise_signal
        total_noise_energy = np.sum(np.square(self.noise_signal))
        remaining_noise_energy = np.sum(np.square(noise))
        noise_removed_percentage = (
            (total_noise_energy - remaining_noise_energy) / total_noise_energy
        ) * 100
        return self.round_to_three_decimal_points(abs(noise_removed_percentage))

    def find_rms(self, signal):
        return np.sqrt(np.mean(np.square(signal)))

    def calculate_rms(self, is_notebook=True):
        rms_clean_signal = self.find_rms(self.clean_signal)
        rms_noise_signal = self.find_rms(self.noise_signal)

        if is_notebook:
            self.plot_rms()
        return {
            "clean_signal": self.round_to_three_decimal_points(rms_clean_signal),
            "noise_signal": self.round_to_three_decimal_points(rms_noise_signal),
        }

    def total_harmonic_distortion(self):
        fundamental_freq_amplitude = np.abs(np.fft.fft(self.clean_signal)[1])
        harmonics_rms = np.sqrt(np.mean(np.square(np.fft.fft(self.noise_signal)[2:])))
        thd = harmonics_rms / fundamental_freq_amplitude
        thd = thd.real
        thd_percent = thd * 100
        return self.round_to_three_decimal_points(thd_percent)

    def auto_correlation(self):
        autocorr = np.correlate(self.clean_signal, self.noise_signal, mode="same")
        autocorr /= np.max(autocorr)
        return autocorr

    def cross_correlation(self):
        crosscorr = np.correlate(self.clean_signal, self.noise_signal, mode="full")
        crosscorr /= np.max(crosscorr)
        return crosscorr

    def plot_rms(self):
        _, ax = plt.subplots(1, 1)
        ax.plot(self.noise_signal, label="Noisy")
        ax.plot(self.clean_signal, label="Clean")
        ax.set_title("RMS Values for Noisy (vs) Denoised")
        ax.set_xlabel(f"Time in {self.time_unit}")
        ax.set_ylabel(f"Amplitude in {self.amplitude_unit}")
        ax.legend()
        plt.show()

    def plot_waves(self, auto_corr, cross_corr):
        plt.figure(figsize=(12, 6))
        plt.subplot(6, 1, 1)
        plt.title("De Noised Signal")
        plt.plot(self.clean_signal, label="Clean Signal")
        plt.xlabel(f"Time in {self.time_unit}")
        plt.ylabel(f"Amplitude in {self.amplitude_unit}")
        plt.legend()

        plt.subplot(6, 1, 3)
        plt.title("Noisy Signal")
        plt.plot(self.noise_signal, label="Noisy Signal")
        plt.xlabel(f"Time in {self.time_unit}")
        plt.ylabel(f"Amplitude in {self.amplitude_unit}")
        plt.legend()

        plt.subplot(6, 1, 5)
        plt.title("Auto-correlation (vs) Cross-correlation")
        plt.plot(auto_corr, label="Auto-correlation")
        plt.plot(cross_corr, label="Cross-correlation")
        plt.xlabel(f"Time in {self.time_unit}")
        plt.ylabel(f"Amplitude in {self.amplitude_unit}")
        plt.legend()
        plt.show()

    def correlation(self, is_notebook=True):
        auto_corr = self.auto_correlation()
        cross_corr = self.cross_correlation()

        if is_notebook:
            self.plot_waves(auto_corr=auto_corr, cross_corr=cross_corr)
        mean_auto_corr = abs(10 * np.log10(np.mean(auto_corr)))
        mean_cross_corr = abs(10 * np.log10(np.mean(cross_corr)))
        return {
            "auto_corr": self.round_to_three_decimal_points(mean_auto_corr),
            "cross_corr": self.round_to_three_decimal_points(mean_cross_corr),
        }

    def round_to_three_decimal_points(self, num):
        return round(num, 3)









