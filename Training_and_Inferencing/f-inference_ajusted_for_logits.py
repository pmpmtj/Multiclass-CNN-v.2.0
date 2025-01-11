import pyaudio
import numpy as np
import torch
import torch.nn as nn
import librosa
from collections import deque
import time

# Parameters
rate = 16000
chunk_size = 3074
record_duration = 1
dynamic_threshold_factor = 0.1
pre_speech_buffer_seconds = 0.8

# VAD Parameters
enable_vad = True
consecutive_frames_required = 3
fixed_threshold = 0.01
low_freq = 300
high_freq = 5000

# Audio Processing Parameters
target_sr = 16000
top_db = 30
target_duration = 1 * target_sr
mfcc_features = 13

# Model Parameters
model_path = "./Training_and_inferencing/multiclass_model.pt"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=rate,
                input=True,
                frames_per_buffer=chunk_size)

# Function to compute energy of audio frame
def compute_energy(frame):
    audio_data = np.frombuffer(frame, dtype=np.int16) / 32768.0
    return np.sum(audio_data ** 2) / len(audio_data)

# Function to apply frequency filtering to audio frame
def apply_frequency_filter(frame):
    audio_data = np.frombuffer(frame, dtype=np.int16) / 32768.0
    fft_data = np.fft.rfft(audio_data)
    freqs = np.fft.rfftfreq(len(audio_data), 1.0 / rate)
    fft_data[(freqs < low_freq) | (freqs > high_freq)] = 0
    filtered_audio = np.fft.irfft(fft_data)
    return np.sum(filtered_audio ** 2) / len(filtered_audio)

# Function to compute dynamic threshold
def compute_dynamic_threshold(duration=5):
    print(f"Calibrating for {duration} seconds to set dynamic threshold...")
    ambient_energy = []
    start_time = time.time()
    while time.time() - start_time < duration:
        audio_frame = stream.read(chunk_size, exception_on_overflow=False)
        energy = compute_energy(audio_frame)
        ambient_energy.append(energy)
    avg_ambient_energy = np.mean(ambient_energy)
    dynamic_threshold = avg_ambient_energy * dynamic_threshold_factor
    print(f"Dynamic threshold set to: {dynamic_threshold:.6f}")
    return dynamic_threshold

# Multi-Class CNN Model
class MultiClassCNN(nn.Module):
    def __init__(self, input_feature_size, num_classes):
        super(MultiClassCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Dynamically calculate the flattened size for fc1
        self.fc1_input_size = self._calculate_flattened_size(input_feature_size)

        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_flattened_size(self, input_feature_size):
        dummy_input = torch.zeros(1, 1, input_feature_size)
        x = self.pool(torch.relu(self.conv1(dummy_input)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output raw logits

# Preprocessing audio using dataset preparation logic
def preprocess_audio(audio_data):
    audio = np.frombuffer(b''.join(audio_data), dtype=np.int16).astype(np.float32)
    audio = librosa.util.normalize(audio)
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    if len(audio_trimmed) > target_duration:
        audio_processed = audio_trimmed[:target_duration]
    else:
        total_padding = target_duration - len(audio_trimmed)
        audio_processed = np.pad(audio_trimmed, (0, total_padding), mode='constant')
    mfcc = librosa.feature.mfcc(y=audio_processed, sr=target_sr, n_mfcc=mfcc_features)
    return mfcc.flatten()

# Main function for VAD and inference
def main():
    if enable_vad:
        threshold = compute_dynamic_threshold()
    else:
        threshold = fixed_threshold

    consecutive_frames = 0
    buffer = deque(maxlen=int(rate / chunk_size * pre_speech_buffer_seconds))

    input_feature_size = mfcc_features * (target_duration // 512 + 1)
    num_classes = 5  # Adjust based on your model's classes
    model = MultiClassCNN(input_feature_size, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    try:
        while True:
            audio_frame = stream.read(chunk_size, exception_on_overflow=False)
            buffer.append(audio_frame)

            if enable_vad:
                energy = apply_frequency_filter(audio_frame)
                if energy > threshold:
                    consecutive_frames += 1
                else:
                    consecutive_frames = 0

                if consecutive_frames >= consecutive_frames_required:
                    print("Speech detected! Capturing audio for inference...")
                    recording_frames = list(buffer)

                    start_time = time.time()
                    while time.time() - start_time < record_duration:
                        audio_frame = stream.read(chunk_size, exception_on_overflow=False)
                        recording_frames.append(audio_frame)

                    features = preprocess_audio(recording_frames)
                    features = torch.tensor(features, dtype=torch.float32).to(device).unsqueeze(0)

                    with torch.no_grad():
                        output = model(features)  # Raw logits
                        probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert to probabilities
                        predicted_class = torch.argmax(probabilities).item()

                    print(f"Predicted class: {predicted_class} (Probabilities: {probabilities})")
                    consecutive_frames = 0
                    buffer.clear()

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
