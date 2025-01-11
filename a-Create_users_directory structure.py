import os

def ensure_directory_exists(directory_path):
    """Ensures a directory exists; creates it if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

def create_word_structure(base_path):
    """
    Prompts the user to enter word names and creates the directory structure for each word.
    """
    words_dir = os.path.join(base_path, "words")
    ensure_directory_exists(words_dir)

    while True:
        word_name = input("Enter the name of the word to add (or type 'q' to quit): ").strip()
        if word_name.lower() == 'q':
            print("Exiting word directory creation.")
            break

        # Create a subdirectory for the word
        word_dir = os.path.join(words_dir, word_name)
        ensure_directory_exists(word_dir)

        # Create the subdirectories within the word directory
        subdirectories = [
            "Approved_Word_Segments",
            "Pre-processed_long_recordings",
            "Un-Approved_Word_Segments"
        ]
        for subdir in subdirectories:
            subdir_path = os.path.join(word_dir, subdir)
            ensure_directory_exists(subdir_path)

        print(f"Structure created for word: {word_name}")

if __name__ == "__main__":
    base_path = input("Enter the base path for your dataset (e.g., C:/Users/yourname/Dataset_Samples): ").strip()
    create_word_structure(base_path)
