import os
import sounddevice as sd
import soundfile as sf
from a_audio_recorder import AudioRecorder
import a_audio_segmenter
from user_management import ensure_directory_exists

    # Configurable path for processed files
DEFAULT_MOVE_PATH = "C:/Processed_Files/test_moving_file"

def record_audio(dataset_dir, username, word):
    """
    Records audio and saves it to the specified word's folder under the user's directory.
    """
    # Initialize recorder
    recorder = AudioRecorder()

    # Construct the save path to the Pre-processed_long_recordings folder for the specific word
    user_word_path = os.path.join(dataset_dir, username, word, "Pre-processed_long_recordings")
    abs_user_word_path = os.path.abspath(user_word_path)  # Convert to absolute path
    ensure_directory_exists(abs_user_word_path)

    # Debug statement to verify the path
    print(f"Saving to directory: {abs_user_word_path}")

    # Find an appropriate filename with a consecutive suffix
    base_filename = username + f'_{word}_recording'
    existing_files = [f for f in os.listdir(abs_user_word_path) if f.startswith(base_filename)]
    next_suffix = len(existing_files) + 1
    filename = f"{base_filename}_{next_suffix}"

    # Start recording
    print("Recording audio... Press Enter to stop.")
    recorder.start_recording()
    input()  # Wait for user input to stop the recording
    recorder.stop_recording()

    # Save the recording
    try:
        recorder.save_recording(abs_user_word_path, filename)
        print(f"Recording saved as '{filename}.wav' in '{abs_user_word_path}'.")
    except Exception as e:
        print(f"Error saving the recording: {e}")

def create_segments(dataset_dir, username, word):
    """Handles creating segments from long audio files for a specific word."""
    input_dir = os.path.join(dataset_dir, username, word, "Pre-processed_long_recordings")
    output_dir = os.path.join(dataset_dir, username, word, "Un-Approved_Word_Segments")
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)

    # Process all audio in the word's directory
    a_audio_segmenter.process_all_audio(input_dir, output_prefix=username + "_" + word + "_segment_", output_dir=output_dir, default_move_path="C:/Processed_Files")


def approve_or_reject_segments(dataset_dir, username, word):
    """
    Handles approving or rejecting audio segments for a specific word, 
    including recursively checking subdirectories.
    """
    import os

    from user_management import ensure_directory_exists

    unapproved_dir = os.path.join(dataset_dir, username, word, "Un-Approved_Word_Segments")
    approved_dir = os.path.join(dataset_dir, username, word, "Approved_Word_Segments")
    ensure_directory_exists(approved_dir)

    # Recursively find all .wav files in unapproved directory and subdirectories
    audio_files = []
    for root, _, files in os.walk(unapproved_dir):
        audio_files.extend([os.path.join(root, f) for f in files if f.lower().endswith('.wav')])

    if not audio_files:
        print(f"No .wav files found in '{unapproved_dir}' or its subdirectories.")
        return

    # List to keep track of files to delete
    files_to_delete = []

    # Loop through each file
    for file_path in audio_files:
        audio_file = os.path.basename(file_path)
        while True:
            print(f"Playing: {audio_file}")
            try:
                data, samplerate = sf.read(file_path)  # Read the audio file
                sd.play(data, samplerate)  # Play the audio
                sd.wait()  # Wait until playback finishes
            except Exception as e:
                print(f"Error playing file {file_path}: {e}")
                break

            # Prompt user for action
            action = input("Do you want to (R) Repeat, (D) Delete, or (C) Continue? ").strip().lower()

            if action == 'r':
                print("Repeating the audio...")
                continue  # Replay the audio

            elif action == 'd':
                print(f"Marking for deletion: {audio_file}")
                files_to_delete.append(file_path)
                break  # Move to the next file

            elif action == 'c':
                # Generate a unique filename in the Approved_Word_Segments folder
                new_path = generate_unique_filename(approved_dir, audio_file, username)

                # Move the file to Approved_Word_Segments
                os.rename(file_path, new_path)
                print(f"Moved '{audio_file}' to '{approved_dir}'.")
                break  # Move to the next file

            else:
                print("Invalid input. Please enter R, D, or C.")

    # Delete all files marked for deletion
    for file_to_delete in files_to_delete:
        try:
            os.remove(file_to_delete)
            print(f"Deleted '{file_to_delete}'.")
        except Exception as e:
            print(f"Error deleting file '{file_to_delete}': {e}")

    print("All files processed.")


def generate_unique_filename(directory, original_filename, username):
    """
    Generate a unique filename in the specified directory to avoid overwriting files.
    """
    base_name, extension = os.path.splitext(original_filename)
    existing_files = [f for f in os.listdir(directory) if f.lower().endswith(extension.lower()) and f.startswith(username)]

    # Extract the numerical suffix from existing filenames and find the next available number
    suffix_numbers = []
    for filename in existing_files:
        try:
            # Extract the number part (e.g., "username_segment_3.wav")
            parts = filename.split('_')
            suffix = int(parts[-1].replace(extension, ''))  # Get the numeric part
            suffix_numbers.append(suffix)
        except ValueError:
            # Ignore files without a numeric suffix or improperly formatted names
            continue

    next_suffix = max(suffix_numbers, default=0) + 1
    new_filename = f"{username}_{base_name}_{next_suffix}{extension}"
    return os.path.join(directory, new_filename)
