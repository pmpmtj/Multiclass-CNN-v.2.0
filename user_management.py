import os

def ensure_directory_exists(dir_path):
    """Ensures a directory exists; creates it if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def list_directories_in_path(path):
    """Lists all directories in the specified path."""
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def create_user(dataset_dir):
    """Handles the 'Create New User' functionality."""
    ensure_directory_exists(dataset_dir)
    
    print("\n*** List of Users in Dataset_Samples ***")
    directories = list_directories_in_path(dataset_dir)
    if directories:
        print("\n".join(directories))
    else:
        print("No directories found.")

    username = input("Enter new username (or 'b' to go back): ").strip()
    if username.lower() == "b":
        return
    
    username = "_".join(username.split())  # Replace spaces with underscores

    user_dir = os.path.join(dataset_dir, username)
    if os.path.exists(user_dir):
        print(f"User '{username}' already exists!")
        return
    
    # Create user directory and 'words' directory
    ensure_directory_exists(user_dir)
    

    print(f"User '{username}' created successfully.")

    # Call create_word_structure to add words
    create_word_structure(user_dir)

def create_word_structure(words_dir):
    """
    Prompts the user to enter word names and creates the directory structure for each word.
    """
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

# if __name__ == "__main__":
#     dataset_dir = input("Enter the base path for your dataset (e.g., C:/Users/yourname/Dataset_Samples): ").strip()
#     create_user(dataset_dir)
