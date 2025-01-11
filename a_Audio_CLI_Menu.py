# # Imports
# # Importing user_management module to handle user-related functionalities like creating and listing users.
import user_management as user_management

# Importing a_audio_processing module for managing audio recording, segmentation, and approval workflows.
import a_audio_processing

# Importing a_cp_app_word for copying approved audio word segments to a specific user.
import a_cp_app_word

# Importing b_add_bckgrd_noise for augmenting audio files with background noise, processing, and dataset management.
import b_add_bckgrd_noise

# Importing os for interacting with the operating system (e.g., path manipulations, clearing the terminal).
import os


def main_menu():
    """
    Handles the main Command Line Interface (CLI) menu for managing audio processing and user data.
    This menu allows users to perform tasks such as creating new users, recording audio, segmenting files,
    approving/rejecting samples, and managing datasets.
    """
    # Directory to store all user data and audio samples.
    dataset_dir = "Dataset_Samples"

    while True:
        # Display the main menu options.
        print("\n*** Main Menu ***")
        print("1. Create New User")  # Add a new user to the dataset directory.
        print("2. New Long Audio Recording - *Add to Existing User*")  # Record long audio for an existing user.
        print("3. Create Segments from Long Audiofiles")  # Segment long audio files into smaller parts.
        print("4. Approve or Reject Segmented Samples")  # Review segmented audio samples for approval/rejection.
        print("5. Copy Approved Words to User")  # Transfer approved audio segments to a user's directory.
        print("6. Augment audiofiles with background noises")  # Add background noises to audio samples.
        print("7. Consolidate audio files to one folder")  # Combine all audio files into a single directory.
        print("8. Process and Extract MFCCs")  # Extract Mel-Frequency Cepstral Coefficients (MFCCs) for audio analysis.
        print("9. Merge, Shuffle and Split into Test and Train Datasets")  # Prepare datasets for model training/testing.
        print("q. Quit")  # Exit the program.

        # Prompt user for their choice.
        choice = input("Enter your choice: ").strip().lower()

        # Option 1: Create a new user and initialize their directories.
        if choice == "1":
            user_management.create_user(dataset_dir)

        # Option 2: Record long audio and save it for an existing user.
        elif choice == "2":
            print("\n*** List of Existing Users ***")
            users = user_management.list_directories_in_path(dataset_dir)

            if not users:
                print("No users found. Please create a user first.")
                continue

            print("\n".join(users))

            username = input("Enter the username to modify (or 'b' to go back): ").strip()
            if username.lower() == "b":
                continue

            if username not in users:
                print(f"User '{username}' does not exist!")
                continue

            user_word_path = os.path.join(dataset_dir, username)
            words = user_management.list_directories_in_path(user_word_path)

            if not words:
                print("No words found for this user. Please create a word first.")
                return

            print("\nAvailable words:")
            print("\n".join(words))

            word = input("Enter the word you want to record for: ").strip()
            if word not in words:
                print(f"Word '{word}' does not exist.")
                continue

            a_audio_processing.record_audio(dataset_dir, username, word)

        # Option 3: Create audio segments from long recordings for an existing user.
        elif choice == "3":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n**WARNING** SEGMENTS HAVE TO BE PROCESSED (approved or rejected) IMMEDIATELY AFTER SEGMENTATION\n")
            print("\n*** List of Existing Users ***")
            users = user_management.list_directories_in_path(dataset_dir)

            if not users:
                print("No users found. Please create a user first.")
                continue

            print("\n".join(users))

            username = input("Enter the username to process long audio files for (or 'b' to go back): ").strip()
            if username.lower() == "b":
                continue

            if username not in users:
                print(f"User '{username}' does not exist!")
                continue

            # List available words for the user
            user_word_path = os.path.join(dataset_dir, username)
            words = user_management.list_directories_in_path(user_word_path)

            if not words:
                print(f"No words found for user '{username}'. Please create words first.")
                continue

            print("\nAvailable words:")
            print("\n".join(words))

            word = input("Enter the word to process long audio files for: ").strip()
            if word not in words:
                print(f"Word '{word}' does not exist.")
                continue

            a_audio_processing.create_segments(dataset_dir, username, word)


        # Option 4: Review segmented samples for approval or rejection.
        elif choice == "4":
            print("\n*** List of Existing Users ***")
            users = user_management.list_directories_in_path(dataset_dir)

            if not users:
                print("No users found. Please create a user first.")
                continue

            print("\n".join(users))

            username = input("Enter the username to approve/reject samples for (or 'b' to go back): ").strip()
            if username.lower() == "b":
                continue

            if username not in users:
                print(f"User '{username}' does not exist!")
                continue

            # List words for the selected user
            user_word_path = os.path.join(dataset_dir, username)
            words = user_management.list_directories_in_path(user_word_path)

            if not words:
                print(f"No words found for user '{username}'. Please create words first.")
                continue

            print("\nAvailable words:")
            print("\n".join(words))

            word = input("Enter the word to approve/reject samples for: ").strip()
            if word not in words:
                print(f"Word '{word}' does not exist.")
                continue

            a_audio_processing.approve_or_reject_segments(dataset_dir, username, word)

            # Clean up empty directories in Un-Approved_Word_Segments
            unapproved_dir = os.path.join(dataset_dir, username, word, "Un-Approved_Word_Segments")
            for root, dirs, _ in os.walk(unapproved_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)


        # Option 5: Copy approved words to a specific user's directory.
        elif choice == "5":
            a_cp_app_word.copy_approved_words_to_user(dataset_dir)


        # Option 6: Add background noises to audio samples.
        elif choice == "6":
            b_add_bckgrd_noise.create_augmented_files(dataset_dir)


        # Option 7: Consolidate audio files into one folder.
        elif choice == "7":
            b_add_bckgrd_noise.consolidate_audio_files(dataset_dir)


        # Option 8: Extract MFCCs from audio samples.
        elif choice == "8":
            b_add_bckgrd_noise.process_and_extract_mfcc(base_dir="./Dataset_Samples", output_csv_dir="./Extract_mfcc")

        # Option 9: Merge datasets, shuffle them, and split into test/train datasets.
        elif choice == "9":
            b_add_bckgrd_noise.merge_and_shuffle_datasets(
                mfcc_dir="./Extract_mfcc",
                output_file="./Shuffle_split_test_train_sets/complete-dataset.csv"
            )
            b_add_bckgrd_noise.split_dataset(
                input_file="./Shuffle_split_test_train_sets/complete-dataset.csv",
                train_file="./Training_and_Inferencing/train_dataset.csv",
                test_file="./Training_and_Inferencing/test_dataset.csv"
            )

        # Option q: Exit the program.
        elif choice in {"q", "quit"}:
            confirm = input("Are you sure you want to quit? (y/n): ").strip().lower()
            if confirm in {"y", "yes"}:
                print("Exiting program. Goodbye!")
                break

        # Handle invalid menu choices.
        else:
            print("Invalid choice. Please try again.")


# Entry point for the program. Ensures the script runs only when executed directly.
if __name__ == "__main__":
    main_menu()
