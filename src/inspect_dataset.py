"""
inspect_dataset.py
------------------
A beginner-friendly utility script to inspect and summarize the structure
of the DeepShip underwater acoustic dataset stored locally on your machine.

Expected Target Classes:
- Cargo
- Passengership
- Tanker
- Tug

Usage:
    python src/inspect_dataset.py --data_dir "C:\\path\\to\\DeepShip"
Or simply run:
    python src/inspect_dataset.py
and you will be prompted to enter your dataset path.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Common audio file extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Expected vessel classes in the DeepShip dataset
EXPECTED_CLASSES = ["Cargo", "Passengership", "Tanker", "Tug"]


def clean_path_input(raw_input: str) -> str:
    """Removes surrounding single/double quotes often added when copying paths on Windows."""
    return raw_input.strip().strip("'").strip('"')


def find_audio_files(directory_path: Path):
    """
    Recursively scans the directory and categorizes audio files by top-level or subfolder class.
    
    Returns:
        dict: {class_name: [list of Path objects]}
        dict: {extension: count}
    """
    class_audio_map = defaultdict(list)
    extension_counts = defaultdict(int)

    # First check direct subdirectories
    subdirs = [p for p in directory_path.iterdir() if p.is_dir()]
    
    if not subdirs:
        # If there are no subdirectories, look for audio files directly in the root
        for file in directory_path.iterdir():
            if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                class_audio_map["Uncategorized"].append(file)
                extension_counts[file.suffix.lower()] += 1
        return class_audio_map, extension_counts

    # Iterate through top-level subdirectories (assumed to be class folders)
    for class_folder in subdirs:
        class_name = class_folder.name
        # Recursively search for audio files inside each class directory
        for file in class_folder.rglob("*"):
            if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                class_audio_map[class_name].append(file)
                extension_counts[file.suffix.lower()] += 1

    return class_audio_map, extension_counts


def inspect_dataset(data_dir_path: str):
    """Inspects the specified folder and prints a detailed, formatted summary."""
    path = Path(data_dir_path)

    print("=" * 65)
    print(" Underwater Acoustic Intrusion Detection - Dataset Inspector")
    print("=" * 65)
    print(f"Target Directory: {path.resolve()}")
    print("-" * 65)

    if not path.exists():
        print(f"\n[ERROR] Directory does not exist: {path.resolve()}")
        print("Please verify the folder path and try again.\n")
        return

    if not path.is_dir():
        print(f"\n[ERROR] The provided path is a file, not a directory: {path.resolve()}\n")
        return

    print("Scanning directory structure for audio files...")
    class_audio_map, extension_counts = find_audio_files(path)

    total_audio_files = sum(len(files) for files in class_audio_map.values())

    if total_audio_files == 0:
        print("\n[WARNING] No supported audio files were found in this directory.")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}")
        print("Subdirectories found:")
        for item in path.iterdir():
            if item.is_dir():
                print(f"  [DIR]  {item.name}/")
            else:
                print(f"  [FILE] {item.name}")
        print("-" * 65)
        return

    # Print File Extension Breakdown
    print("\n[+] File Formats Found:")
    for ext, count in sorted(extension_counts.items()):
        print(f"  * {ext}: {count} file(s)")

    # Print Class Breakdown
    print("\n[+] Category / Class Breakdown:")
    print(f"{'Class Name':<20} | {'Count':<10} | {'Status'}")
    print("-" * 55)

    detected_classes = set(class_audio_map.keys())

    # Display expected classes first
    for exp_class in EXPECTED_CLASSES:
        # Match case-insensitively
        matched_class = next((c for c in detected_classes if c.lower() == exp_class.lower()), None)
        if matched_class:
            count = len(class_audio_map[matched_class])
            status = "[OK] Found" if count > 0 else "[WARN] Empty"
            print(f"{matched_class:<20} | {count:<10} | {status}")
        else:
            print(f"{exp_class:<20} | {0:<10} | [MISSING] Folder not found")

    # Display any extra directories found that weren't in the 4 expected classes
    extra_classes = [
        c for c in detected_classes 
        if not any(c.lower() == exp.lower() for exp in EXPECTED_CLASSES)
    ]
    if extra_classes:
        print("-" * 55)
        print("Additional Folders Found:")
        for extra in extra_classes:
            count = len(class_audio_map[extra])
            print(f"{extra:<20} | {count:<10} | [INFO] Other category")

    print("-" * 55)
    print(f"{'Total Audio Files:':<20} | {total_audio_files:<10}")
    print("=" * 65)

    # Sample File Previews
    print("\n[+] Sample Audio Files (up to 2 per class):")
    for cls_name, files in sorted(class_audio_map.items()):
        print(f"\n[{cls_name}] ({len(files)} total):")
        for sample in files[:2]:
            print(f"  |-- {sample.name} (in {sample.parent.name})")

    print("\n" + "=" * 65)
    print("Inspection complete. Dataset is ready for the next preprocessing phase.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and summarize audio files in your local DeepShip dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the root folder of your DeepShip dataset.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        print("=" * 65)
        print(" Underwater Acoustic Intrusion Detection - Dataset Inspector")
        print("=" * 65)
        try:
            data_dir = input("\nPlease enter the full path to your dataset folder: ")
        except (KeyboardInterrupt, EOFError):
            print("\nInspection cancelled.")
            sys.exit(0)

    data_dir = clean_path_input(data_dir)
    if not data_dir:
        print("[ERROR] No path entered. Exiting.")
        sys.exit(1)

    inspect_dataset(data_dir)


if __name__ == "__main__":
    main()
