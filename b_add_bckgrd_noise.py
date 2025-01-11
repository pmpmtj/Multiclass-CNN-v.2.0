# Import necessary libraries
import os
import random
from pydub import AudioSegment
from pydub.playback import play
import json
import shutil
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_noise_files(noise_root):
    """
    Recursively fetch all noise files from the given root directory.
    """
    noise_files = []
    for root, _, files in os.walk(noise_root):
        noise_files.extend([os.path.join(root, f) for f in files if f.endswith('.wav')])
    return noise_files


def overlay_audio(source_audio, noise_audio, noise_volume):
    """
    Overlay noise on source audio with specified noise volume adjustment.
    """
    noise_audio = noise_audio - noise_volume  # Adjust noise volume
    if len(noise_audio) < len(source_audio):
        noise_audio = noise_audio * (len(source_audio) // len(noise_audio) + 1)  # Loop noise to match duration
    noise_audio = noise_audio[:len(source_audio)]  # Trim noise to match source duration
    return source_audio.overlay(noise_audio)


def process_audio_with_noise(user_path, noise_root, config):
    """
    Process approved audio files by adding background noises.
    """
    source_files = [
        os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith('.wav')
    ]
    if not source_files:
        print(f"No source audio files found in {user_path}.")
        return

    noise_files = get_noise_files(noise_root)
    if not noise_files:
        print(f"No noise audio files found in {noise_root}.")
        return

    # Load configuration settings
    noise_volume = config.get('noise_volume', 10)  # Default volume reduction of 10 dB

    current_file_count = len(source_files)
    new_file_index = current_file_count + 1

    for source_file in source_files:
        source_audio = AudioSegment.from_file(source_file)

        for noise_file in noise_files:
            noise_audio = AudioSegment.from_file(noise_file)

            # Overlay source and noise
            augmented_audio = overlay_audio(source_audio, noise_audio, noise_volume)

            # Save the augmented file
            output_file = os.path.join(
                user_path, f"augmented_{new_file_index}.wav"
            )
            augmented_audio.export(output_file, format="wav")
            print(f"Saved augmented file: {output_file}")

            new_file_index += 1


def consolidate_audio_files(dataset_dir):
    """
    Automatically consolidates all words (classes) from all users into separate unified directories.
    """
    # Get all users in the dataset directory
    users = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not users:
        print("No users found. Please create a user first.")
        return

    # Dictionary to keep track of global file counts for each word/class
    global_file_counts = {}

    def consolidate(source_dir, output_dir, word, global_file_count):
        """
        Consolidates audio files from the source directory to the output directory.
        """
        file_count = global_file_count
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_count += 1
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(output_dir, f"{word}_{file_count}.wav")
                    shutil.copy2(src_path, dest_path)
                    print(f"Copied {src_path} -> {dest_path}")
        return file_count

    # Iterate over each user
    for user in users:
        user_path = os.path.join(dataset_dir, user)
        if not os.path.isdir(user_path):
            continue

        # List all words (classes) for the user
        words = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
        for word in words:
            word_source_dir = os.path.join(user_path, word, "Approved_Word_Segments")

            if os.path.exists(word_source_dir):
                # Create a "joined" directory for the word/class
                joined_word_dir = os.path.join(dataset_dir, f"joined_{word}")
                os.makedirs(joined_word_dir, exist_ok=True)

                # Initialize file count for the word if not already done
                if word not in global_file_counts:
                    global_file_counts[word] = 0

                # Consolidate files for the word/class
                global_file_counts[word] = consolidate(
                    word_source_dir,
                    joined_word_dir,
                    word,
                    global_file_counts[word]
                )

    print("Consolidation complete.")

def process_and_extract_mfcc(base_dir, output_csv_dir, target_sr=16000, top_db=30, target_duration_seconds=1, mfcc_features=13):
    """
    Extract MFCC features from consolidated audio files for multiple classes.
    Ensures no redundant columns and numerical labels are added correctly.
    """
    target_duration = target_duration_seconds * target_sr

    def normalize_audio(audio):
        """Normalize audio to the range [-1, 1]."""
        if np.max(np.abs(audio)) > 0:
            return audio / np.max(np.abs(audio))
        return audio

    def process_audio(audio):
        """Trim and pad audio to a fixed duration."""
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        audio_normalized = normalize_audio(audio_trimmed)
        if len(audio_normalized) > target_duration:
            return audio_normalized[:target_duration]
        return np.pad(audio_normalized, (0, target_duration - len(audio_normalized)), mode="constant")

    def extract_mfcc(input_dir, label, output_csv):
        """Extract MFCC features from audio files in the input directory."""
        mfcc_data = []
        for file in os.listdir(input_dir):
            if file.endswith('.wav'):
                try:
                    path = os.path.join(input_dir, file)
                    audio, sr = librosa.load(path, sr=target_sr)
                    processed_audio = process_audio(audio)
                    mfcc = librosa.feature.mfcc(y=processed_audio, sr=sr, n_mfcc=mfcc_features).flatten()
                    mfcc_data.append(np.append(mfcc, label))  # Append numerical label
                except Exception as e:
                    print(f"Error processing {file}: {e}")
        if mfcc_data:
            df = pd.DataFrame(mfcc_data, columns=[f'feature_{i}' for i in range(len(mfcc_data[0]) - 1)] + ['label'])
            df.to_csv(output_csv, index=False)  # Save only numerical labels

    os.makedirs(output_csv_dir, exist_ok=True)

    # Dynamically find all "joined_<word>" directories
    for idx, folder in enumerate(os.listdir(base_dir)):
        if folder.startswith("joined_"):
            word = folder[len("joined_"):]  # Extract the word name
            input_dir = os.path.join(base_dir, folder)
            output_csv = os.path.join(output_csv_dir, f"mfcc_{word}.csv")

            print(f"Processing MFCC for word: {word} (Label: {idx})")
            extract_mfcc(input_dir, label=idx, output_csv=output_csv)  # Use numerical labels (e.g., 0, 1, 2)

    print("MFCC extraction for all classes is complete.")



def merge_and_shuffle_datasets(mfcc_dir, output_file):
    """
    Merge multiple class datasets (MFCC features), shuffle, and save as a single CSV.
    Removes redundant columns and ensures a clean dataset.
    """
    mfcc_files = [f for f in os.listdir(mfcc_dir) if f.startswith("mfcc_") and f.endswith(".csv")]
    if not mfcc_files:
        print(f"No MFCC CSV files found in directory '{mfcc_dir}'.")
        return

    class_mapping = {}
    merged_df = pd.DataFrame()

    for idx, mfcc_file in enumerate(mfcc_files):
        class_label = os.path.splitext(mfcc_file)[0].replace("mfcc_", "")
        file_path = os.path.join(mfcc_dir, mfcc_file)

        try:
            df = pd.read_csv(file_path)

            # Assign numerical labels
            if class_label not in class_mapping:
                class_mapping[class_label] = idx

            df['label'] = class_mapping[class_label]

            # Drop any extra or redundant columns
            if 'label' in df.columns[:-1]:  # Ensure only the last column is 'label'
                df = df.drop(columns=[col for col in df.columns[:-1] if col != 'label'])

            merged_df = pd.concat([merged_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")

    # Shuffle and save
    shuffled_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_df.to_csv(output_file, index=False)
    print(f"Merged and shuffled dataset saved to {output_file}")
    print(f"Class Mapping: {class_mapping}")





def split_dataset(input_file, train_file, test_file, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets and save to separate CSV files.
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Split into training and testing sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Save the splits
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Training set saved to {train_file}")
    print(f"Testing set saved to {test_file}")


# Main entry point for menu options
def create_augmented_files(dataset_dir):
    """
    Adds background noises to approved audio samples in the user's word directory.
    """
    users = [
        d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    if not users:
        print("No users found. Please create a user first.")
        return

    print("\n*** List of Existing Users ***")
    print("\n".join(users))

    username = input("Enter the username to augment audio files for (or 'b' to go back): ").strip()
    if username.lower() == "b":
        return

    if username not in users:
        print(f"User '{username}' does not exist!")
        return

    # List words for the selected user
    user_word_path = os.path.join(dataset_dir, username)
    words = [
        d for d in os.listdir(user_word_path) if os.path.isdir(os.path.join(user_word_path, d))
    ]
    if not words:
        print(f"No words found for user '{username}'. Please create words first.")
        return

    print("\nAvailable words:")
    print("\n".join(words))

    word = input("Enter the word to augment audio files for: ").strip()
    if word not in words:
        print(f"Word '{word}' does not exist!")
        return

    user_path = os.path.join(user_word_path, word, "Approved_Word_Segments")
    if not os.path.exists(user_path):
        print(f"The directory '{user_path}' does not exist!")
        return

    # Check for configuration file
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Configuration file '{config_path}' is missing.")
        return

    # Load configuration
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Check for noise directory
    noise_root = config.get("noise_directory", "./noise_files_directory")
    if not os.path.exists(noise_root):
        print(f"Noise directory '{noise_root}' does not exist!")
        return

    # Process the audio files with noise
    process_audio_with_noise(user_path, noise_root, config)
    print("All files have been processed.")



# To integrate with the CLI menu, add the following:
# elif choice == "6":
#     create_augmented_files(dataset_dir)
# elif choice == "7":
#     consolidate_audio_files(dataset_dir)
# elif choice == "8":
#     process_and_extract_mfcc(base_dir="./Dataset_Samples", output_csv_dir="./Extract_mfcc")
# elif choice == "9":
#     merge_and_shuffle_datasets(
#         callword_csv="./Extract_mfcc/mfcc_callword.csv",
#         non_callword_csv="./Extract_mfcc/mfcc_non_callword.csv",
#         output_file="./Shuffle_split_test_train_sets/complete-dataset.csv"
#     )
#     split_dataset(
#         input_file="./Shuffle_split_test_train_sets/complete-dataset.csv",
#         train_file="./Training_and_Inferencing/train_dataset.csv",
#         test_file="./Training_and_Inferencing/test_dataset.csv"
#     )
