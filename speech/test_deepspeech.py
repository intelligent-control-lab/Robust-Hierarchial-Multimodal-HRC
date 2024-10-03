import deepspeech
import wave
import numpy as np
from pathlib import Path
import pyaudio
FILE_DIR = Path(__file__).parent
import matplotlib.pyplot as plt
import pdb
# Initialize the DeepSpeech model with the downloaded model and scorer
model = deepspeech.Model(f'{FILE_DIR}/deepspeech-0.9.3-models.pbmm')
model.enableExternalScorer(f'{FILE_DIR}/deepspeech-0.9.3-models.scorer')

import noisereduce as nr
import soundfile as sf

# Define parameters
sample_rate = 16000  # Adjust as needed
chunk_size = 1024

# Create a PyAudio stream for microphone input
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                output=True,  # Enable audio output
                frames_per_buffer=chunk_size)

# Initialize a noise profile with the first few seconds of input
noise_profile = np.zeros(chunk_size, dtype=np.float32)
for _ in range(10):  # Capture the first 10 chunks of audio as a noise profile
    audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
    noise_profile += audio_data

noise_profile = noise_profile / 10  # Average the noise profile
ds_stream = model.createStream()
print("Noise profile captured. Start speaking.")

try:
    count = 0
    init_audio = np.array([])
    denoised_audio = np.array([])  # Initialize an empty array to collect denoised audio
    while True:
        audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
        init_audio = np.append(init_audio,audio_data)
        # Apply noise reduction using the noise profile
        denoised_chunk = nr.reduce_noise(y=audio_data, sr=sample_rate)
        denoised_audio = np.append(denoised_audio, denoised_chunk)
        # Process the denoised audio data as needed
        
        ds_stream.feedAudioContent(denoised_chunk)
        partial_transcription = ds_stream.intermediateDecode()
        if len(partial_transcription) > count:
            count = len(partial_transcription)
            print(partial_transcription.split()[-1])

except KeyboardInterrupt:
    print("Stopped listening.")

plt.plot(init_audio)
plt.savefig(f'{FILE_DIR}/init_audio.jpg')
plt.close()
# Compute the FFT
fft_result = np.fft.fft(denoised_audio)

# Compute the corresponding frequencies
num_samples = len(denoised_audio)
frequencies = np.fft.fftfreq(num_samples, 1.0 / sample_rate)
# pdb.set_trace()
# Compute the magnitude of the FFT result
magnitude = np.abs(fft_result)

# Apply a low-pass filter (simple moving average)
high_cutoff_frequency = 800  # Adjust the cutoff frequency as needed
low_cutoof_ferquency = 700
mask1 = abs(frequencies) < high_cutoff_frequency
mask2 = abs(frequencies) > low_cutoof_ferquency
mask = ~(mask1 * mask2)
# pdb.set_trace()
high_cutoff_frequency = 1600  # Adjust the cutoff frequency as needed
low_cutoof_ferquency = 1300
mask1 = abs(frequencies) < high_cutoff_frequency
mask2 = abs(frequencies) > low_cutoof_ferquency
mask = ~(mask1 * mask2) * mask
high_cutoff_frequency = 2000
low_cutoof_ferquency = 50
mask1 = abs(frequencies) <= high_cutoff_frequency
mask2 = abs(frequencies) >= low_cutoof_ferquency
mask = mask1 * mask

# mask = mask1
# pdb.set_trace()
# Apply the filter to the magnitude array
# filtered_fft_result = np.copy(fft_result)
filtered_fft_result = fft_result * mask

# Inverse FFT to obtain the filtered signal
# pdb.set_trace()
filtered_audio_data = np.fft.ifft(filtered_fft_result).real

# Plot the magnitude vs. frequency
plt.figure(figsize=(10, 4))
plt.plot(frequencies[:num_samples // 2], magnitude[:num_samples // 2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.grid(True)
plt.savefig(f'{FILE_DIR}/init_audio_fft.jpg')
plt.close()
# Close the PyAudio stream
stream.stop_stream()
stream.close()
p.terminate()

plt.figure(figsize=(10, 4))
plt.plot(frequencies[:num_samples // 2], np.abs(filtered_fft_result)[:num_samples // 2])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum")
plt.grid(True)
plt.savefig(f'{FILE_DIR}/filtered_audio_fft.jpg')
plt.close()


print("Real-time noise reduction and transcription completed.")

# Save the denoised audio as a WAV file
sf.write('init_audio.wav',init_audio,sample_rate)
sf.write('denoised_audio.wav', denoised_audio, sample_rate)
sf.write('filtered_audio.wav', filtered_audio_data, sample_rate)
print("Denoised audio saved as denoised_audio.wav")
