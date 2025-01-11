from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import glob
import shutil

# ===== CONFIGURATION VARIABLES =====
MIN_SILENCE_LEN = 250                          # Minimum length of silence to consider as a split point (in ms)250
SILENCE_THRESH = -40                     # Silence threshold in dB -40, 
TARGET_DURATION = 1000                         # Target duration for each segment in milliseconds (1 second)
# ===== END OF CONFIGURATION =====

def load_audio(file_path):
    """Load an audio file into an AudioSegment object."""
    try:
        return AudioSegment.from_file(file_path)
    except FileNotFoundError:
        print(f"Error: The input file '{file_path}' does not exist. Please check the file path and try again.")
        return None

def split_audio_on_silence(audio, min_silence_len=MIN_SILENCE_LEN, silence_thresh=SILENCE_THRESH):
    """Split an audio file into segments based on silence."""
    return split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

def adjust_duration(segment, target_duration=TARGET_DURATION):
    """Adjust the duration of a segment by trimming or padding with silence."""
    if len(segment) > target_duration:
        segment = segment[:target_duration]
    elif len(segment) < target_duration:
        padding_duration = target_duration - len(segment)
        silence_padding = AudioSegment.silent(duration=padding_duration)
        segment = segment + silence_padding
    return segment

def ensure_directory_exists(directory_path):
    """Ensure the output directory exists, or create it if it doesn't."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def save_audio_segments(segments, output_prefix, output_dir, start_index):
    """Save audio segments as separate .wav files."""
    ensure_directory_exists(output_dir)

    for i, segment in enumerate(segments, start=start_index):
        adjusted_segment = adjust_duration(segment)
        file_name = os.path.join(output_dir, f"{output_prefix}{i}.wav")
        try:
            adjusted_segment.export(file_name, format="wav")
            print(f"Saved segment {i} as '{file_name}'")
        except Exception as e:
            print(f"Error saving file '{file_name}': {e}")

def process_all_audio(input_dir, output_prefix, output_dir, start_index=1, default_move_path="C:/Processed_Files"):
    """
    Process all .wav files in the input directory by splitting them on silence,
    saving the segments, and optionally moving the processed file to another location.
    """
    if not os.path.exists(input_dir):
        print(f"Error: The input directory '{input_dir}' does not exist.")
        return

    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in the directory '{input_dir}'.")
        return

    for file_path in wav_files:
        print(f"Processing file: {file_path}")
        audio = load_audio(file_path)
        if audio is None:
            continue

        segments = split_audio_on_silence(audio)
        if segments:
            # Create subfolder with the same name as the filename (without extension)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            file_output_dir = os.path.join(output_dir, file_name)
            ensure_directory_exists(file_output_dir)
            save_audio_segments(segments, output_prefix, file_output_dir, start_index)
        else:
            print(f"No segments found to save for file '{file_path}'. Please check the silence threshold and other parameters.")
        
        # Prompt the user about moving the processed file
        move_processed_file(file_path, default_move_path)


def move_processed_file(file_path, default_path):
    """
    Move a processed file to a specified location.
    Prompts the user for confirmation and optional path input.
    """
    while True:
        move_file = input("Do you want to move the processed file to another location? (yes/no) [default: yes]: ").strip().lower()
        if move_file in {"", "yes", "y"}:
            print(f"\nDefault path: {default_path}")
            use_default = input("Do you want to use the default path? (yes/no) [default: yes]: ").strip().lower()
            if use_default in {"", "yes", "y"}:
                target_path = default_path
            else:
                target_path = input("Enter the preferred path to move the file to: ").strip()
            
            # Normalize the path for Windows-style input
            target_path = os.path.normpath(target_path)

            # Ensure target directory exists
            os.makedirs(target_path, exist_ok=True)

            # Move the file
            try:
                shutil.move(file_path, target_path)
                print(f"File moved successfully to '{target_path}'.")
            except Exception as e:
                print(f"Error moving file: {e}")
            break
        elif move_file in {"no", "n"}:
            print("Skipping file move.")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
