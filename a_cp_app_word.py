import os
import shutil

def copy_approved_words_to_user(dataset_dir):
    """
    Handles copying approved words from an external directory to the user's Approved_Word_Segments folder.
    """
    if not os.path.exists(dataset_dir):
        print("No users found. Please create a user first.")
        return

    # List existing users
    print("\n*** List of Existing Users ***")
    users = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not users:
        print("No users found. Please create a user first.")
        return
    print("\n".join(users))

    # Prompt for username
    username = input("Enter the username to copy approved words for (or 'b' to go back): ").strip()
    if username.lower() == "b":
        return
    if username not in users:
        print(f"User '{username}' does not exist!")
        return

    # List words for the selected user
    user_word_path = os.path.join(dataset_dir, username)
    words = [d for d in os.listdir(user_word_path) if os.path.isdir(os.path.join(user_word_path, d))]
    if not words:
        print(f"No words found for user '{username}'. Please create words first.")
        return

    print("\nAvailable words:")
    print("\n".join(words))

    word = input("Enter the word to copy approved files for: ").strip()
    if word not in words:
        print(f"Word '{word}' does not exist!")
        return

    # Set the destination path
    approved_dir = os.path.join(user_word_path, word, "Approved_Word_Segments")
    os.makedirs(approved_dir, exist_ok=True)

    # Prompt for source directory (external directory containing approved words)
    source_dir = input("Enter the absolute path for the source directory (e.g., c:\\users\\username\\source_folder): ").strip()
    source_dir = os.path.normpath(source_dir)  # Normalize the path for cross-platform compatibility

    # Validate the source directory
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    # Get all .wav files in the source directory
    audio_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.wav')]
    if not audio_files:
        print(f"No .wav files found in the directory '{source_dir}'.")
        return

    # Get the highest suffix number in the destination directory to avoid overwriting
    existing_files = [f for f in os.listdir(approved_dir) if f.lower().endswith('.wav') and f.startswith(username)]
    suffix_numbers = []
    for filename in existing_files:
        try:
            parts = filename.split('_')
            suffix = int(parts[-1].replace('.wav', ''))  # Get the numeric part
            suffix_numbers.append(suffix)
        except ValueError:
            continue

    # Start numbering from the highest existing suffix + 1
    next_suffix = max(suffix_numbers, default=0) + 1

    # Copy each .wav file with a unique name
    for audio_file in audio_files:
        source_file_path = os.path.join(source_dir, audio_file)

        # Generate a unique filename
        new_filename = f"{username}_{word}_segment_{next_suffix}.wav"
        destination_file_path = os.path.join(approved_dir, new_filename)

        # Copy the file
        try:
            shutil.copy2(source_file_path, destination_file_path)
            #print(f"Copied '{audio_file}' to '{destination_file_path}'.")
        except Exception as e:
            print(f"Error copying file '{audio_file}': {e}")

        # Increment suffix for next file
        next_suffix += 1

    print("All files have been copied successfully.")


