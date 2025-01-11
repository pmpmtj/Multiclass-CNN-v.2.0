import wave
import threading
import pyaudio
import os
import time

# ===== CONFIGURATION VARIABLES =====
# Audio settings
AUDIO_FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
AUDIO_CHANNELS = 1              # Mono audio
AUDIO_RATE = 16000              # Sample rate (Hz)
AUDIO_INPUT = True              # Enable audio input
AUDIO_FRAMES_PER_BUFFER = 1024  # Buffer size for audio stream
TRIM_LAST_MS = 0.05             # Trimming duration at the end in seconds
TRIM_FIRST_MS = 0.05            # Trimming duration at the beginning in seconds
# ===== END OF CONFIGURATION =====

class AudioRecorder:
    """
    A class for recording, pausing, and saving audio input using the pyaudio library.
    """
    def __init__(self):
        self.frames = []
        self.is_recording = False
        self.is_paused = False
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.thread = None
        self.start_time = None  # Track the start time of recording
        self.elapsed_time = 0   # Track the total elapsed time

    def start_recording(self):
        """Starts audio recording in a separate thread."""
        self.frames = []
        self.is_recording = True
        self.is_paused = False
        self.start_time = time.time()
        self.elapsed_time = 0
        self.stream = self.audio.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=AUDIO_INPUT,
            frames_per_buffer=AUDIO_FRAMES_PER_BUFFER
        )
        self.thread = threading.Thread(target=self.record)
        self.thread.start()

    def record(self):
        """Continuously records audio data while recording is active."""
        while self.is_recording:
            if not self.is_paused:
                data = self.stream.read(AUDIO_FRAMES_PER_BUFFER)
                self.frames.append(data)
                self.elapsed_time = time.time() - self.start_time

    def pause_recording(self):
        """Toggles pause state and updates elapsed time."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.trim_last_ms()

    def stop_recording(self):
        """Stops the recording and releases resources."""
        self.is_recording = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.trim_last_ms()

    def trim_first_ms(self):
        """Trims the first few milliseconds of audio data if TRIM_FIRST_MS is greater than 0."""
        if TRIM_FIRST_MS > 0:
            samples_per_ms = int(AUDIO_RATE * TRIM_FIRST_MS)
            bytes_per_sample = 2  # 16-bit audio = 2 bytes per sample
            bytes_per_ms = samples_per_ms * bytes_per_sample
            frames_to_trim = bytes_per_ms // AUDIO_FRAMES_PER_BUFFER
            self.frames = self.frames[frames_to_trim:]

    def trim_last_ms(self):
        """Trims the last few milliseconds of audio data to ensure clean cuts."""
        samples_per_ms = int(AUDIO_RATE * TRIM_LAST_MS)
        bytes_per_sample = 2  # 16-bit audio = 2 bytes per sample
        bytes_per_ms = samples_per_ms * bytes_per_sample
        if len(self.frames) > bytes_per_ms // AUDIO_FRAMES_PER_BUFFER:
            self.frames = self.frames[:-bytes_per_ms // AUDIO_FRAMES_PER_BUFFER]

    def save_recording(self, save_path, filename):
        """Saves the recorded audio to a .wav file, trimming the beginning if needed."""
        self.trim_first_ms()  # Trim the beginning of the recording before saving
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename + '.wav')
        wf = wave.open(full_path, 'wb')
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(AUDIO_FORMAT))
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        return full_path  # Return the full path of the saved file
