"""
analyze_audio.py
----------------
Comprehensive dataset inspection and audio metadata analysis for DeepShip.

Scans the DeepShip dataset directory, validates audio integrity, checks for
corrupt or empty files, extracts sample rates, channel counts, durations,
and bit depths, calculates per-class duration statistics, and exports
verified metadata to CSV and JSON reports in outputs/.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import soundfile as sf
import pandas as pd
import numpy as np

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, ensure_dir, save_json, CANONICAL_CLASSES, get_project_root


def inspect_audio_file(file_path: Path, expected_class: str) -> Dict[str, Any]:
    """
    Safely inspects a single audio file without modifying it.
    
    Returns:
        dict containing audio technical metadata and integrity status.
    """
    record = {
        "file_name": file_path.name,
        "relative_path": str(file_path),
        "class_name": expected_class,
        "file_size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        "sample_rate": None,
        "channels": None,
        "frames": 0,
        "duration_seconds": 0.0,
        "subtype": None,
        "format": None,
        "is_valid": False,
        "error_message": None,
    }

    if record["file_size_bytes"] == 0:
        record["error_message"] = "Empty file (0 bytes)"
        return record

    try:
        # Use soundfile SoundFile inspection (very fast, does not decode whole audio)
        with sf.SoundFile(str(file_path)) as audio_info:
            record["sample_rate"] = int(audio_info.samplerate)
            record["channels"] = int(audio_info.channels)
            record["frames"] = int(audio_info.frames)
            record["duration_seconds"] = float(audio_info.frames / audio_info.samplerate) if audio_info.samplerate > 0 else 0.0
            record["subtype"] = audio_info.subtype
            record["format"] = audio_info.format

            if record["frames"] <= 0 or record["duration_seconds"] <= 0:
                record["error_message"] = "Zero frames or non-positive duration"
            else:
                record["is_valid"] = True

    except Exception as e:
        record["error_message"] = str(e)

    return record


def scan_and_analyze_dataset(data_dir: str or Path) -> pd.DataFrame:
    """
    Recursively scans data_dir for all canonical classes and extracts audio metadata.
    """
    root_path = Path(data_dir)
    if not root_path.exists() or not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root directory does not exist or is not a directory: {root_path.resolve()}")

    records = []
    print(f"\n[+] Scanning dataset at: {root_path.resolve()}")

    for class_name in CANONICAL_CLASSES:
        # Search case-insensitively for class folder
        matching_dirs = [d for d in root_path.iterdir() if d.is_dir() and d.name.lower() == class_name.lower()]

        if not matching_dirs:
            print(f"  [!] Missing class folder for: {class_name}")
            continue

        class_dir = matching_dirs[0]
        # Recursively search for .wav files
        wav_files = sorted(list(class_dir.rglob("*.wav")))
        print(f"  Found {len(wav_files)} WAV file(s) for class '{class_name}'")

        for wav_path in wav_files:
            rec = inspect_audio_file(wav_path, class_name)
            # Make path relative to data_dir for portable storage
            try:
                rec["relative_to_data_root"] = str(wav_path.relative_to(root_path))
            except ValueError:
                rec["relative_to_data_root"] = wav_path.name
            records.append(rec)

    df = pd.DataFrame(records)
    return df


def generate_analysis_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates a structured summary dictionary of audio statistics."""
    summary = {
        "total_recordings_scanned": int(len(df)),
        "valid_recordings": int(df["is_valid"].sum()) if not df.empty else 0,
        "corrupted_or_empty": int((~df["is_valid"]).sum()) if not df.empty else 0,
        "classes_detected": sorted(df["class_name"].unique().tolist()) if not df.empty else [],
        "sample_rate_breakdown": {},
        "channel_breakdown": {},
        "subtype_breakdown": {},
        "class_statistics": {},
    }

    if df.empty or summary["valid_recordings"] == 0:
        return summary

    valid_df = df[df["is_valid"]]

    # Sample rates & channels
    summary["sample_rate_breakdown"] = {str(k): int(v) for k, v in valid_df["sample_rate"].value_counts().items()}
    summary["channel_breakdown"] = {str(k): int(v) for k, v in valid_df["channels"].value_counts().items()}
    summary["subtype_breakdown"] = {str(k): int(v) for k, v in valid_df["subtype"].value_counts().items()}

    # Per class duration stats
    for cls_name in CANONICAL_CLASSES:
        cls_df = valid_df[valid_df["class_name"] == cls_name]
        count = len(cls_df)
        if count > 0:
            durations = cls_df["duration_seconds"]
            summary["class_statistics"][cls_name] = {
                "file_count": count,
                "total_duration_seconds": round(float(durations.sum()), 2),
                "total_duration_minutes": round(float(durations.sum() / 60.0), 2),
                "mean_duration_seconds": round(float(durations.mean()), 2),
                "median_duration_seconds": round(float(durations.median()), 2),
                "min_duration_seconds": round(float(durations.min()), 2),
                "max_duration_seconds": round(float(durations.max()), 2),
            }
        else:
            summary["class_statistics"][cls_name] = {
                "file_count": 0,
                "status": "Missing or no valid audio files"
            }

    summary["total_audio_duration_seconds"] = round(float(valid_df["duration_seconds"].sum()), 2)
    summary["total_audio_duration_hours"] = round(float(valid_df["duration_seconds"].sum() / 3600.0), 2)
    return summary


def print_analysis_report(summary: Dict[str, Any], metadata_csv_path: Path):
    """Prints a clear, formatted terminal report."""
    print("\n" + "=" * 70)
    print(" DEEPSHIP UNDERWATER AUDIO DATASET ANALYSIS REPORT")
    print("=" * 70)
    print(f"Total Audio Files Scanned : {summary['total_recordings_scanned']}")
    print(f"Valid Intact Recordings   : {summary['valid_recordings']}")
    print(f"Corrupted or Empty Files  : {summary['corrupted_or_empty']}")
    print(f"Total Duration (Hours)    : {summary.get('total_audio_duration_hours', 0.0)} hrs")
    print("-" * 70)

    print("\n[+] Sample Rate & Channel Consistency:")
    for sr, cnt in summary.get("sample_rate_breakdown", {}).items():
        print(f"  * Sample Rate {sr} Hz: {cnt} file(s)")
    for ch, cnt in summary.get("channel_breakdown", {}).items():
        ch_str = "Mono (1 ch)" if ch == "1" else f"{ch} channels"
        print(f"  * Channels: {ch_str}: {cnt} file(s)")
    for st, cnt in summary.get("subtype_breakdown", {}).items():
        print(f"  * Audio Encoding Subtype: {st} ({cnt} files)")

    print("\n[+] Per-Class Audio Breakdown & Duration Statistics:")
    print(f"{'Class':<15} | {'Files':<6} | {'Total (min)':<12} | {'Mean (s)':<10} | {'Min (s)':<8} | {'Max (s)'}")
    print("-" * 70)
    for cls_name, stats in summary.get("class_statistics", {}).items():
        if stats.get("file_count", 0) > 0:
            print(
                f"{cls_name:<15} | {stats['file_count']:<6} | "
                f"{stats['total_duration_minutes']:<12.1f} | "
                f"{stats['mean_duration_seconds']:<10.1f} | "
                f"{stats['min_duration_seconds']:<8.1f} | "
                f"{stats['max_duration_seconds']:.1f}"
            )
        else:
            print(f"{cls_name:<15} | 0      | 0.0          | 0.0        | 0.0      | 0.0 (EMPTY)")

    print("-" * 70)
    print(f"[+] Verified audio metadata saved to: {metadata_csv_path.resolve()}")
    print("=" * 70 + "\n")


def run_analysis(data_dir: Optional[str] = None, output_csv: Optional[str] = None):
    """Main entrypoint for dataset analysis."""
    config = load_config()
    target_data_dir = data_dir or config["dataset"]["raw_data_dir"]
    target_out_csv = Path(output_csv or (get_project_root() / config["paths"]["metadata_csv"]))

    ensure_dir(target_out_csv.parent)

    df = scan_and_analyze_dataset(target_data_dir)
    if df.empty:
        print("[ERROR] No recordings found. Please check dataset path.")
        return df, {}

    # Save CSV
    df.to_csv(target_out_csv, index=False)

    # Save JSON summary report
    summary = generate_analysis_summary(df)
    report_json_path = target_out_csv.parent / "dataset_inspection_report.json"
    save_json(summary, report_json_path)

    print_analysis_report(summary, target_out_csv)
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DeepShip audio metadata and data integrity.")
    parser.add_argument("--data_dir", type=str, default=None, help="Root folder of DeepShip dataset")
    parser.add_argument("--output_csv", type=str, default=None, help="Output path for metadata CSV")
    args = parser.parse_args()

    run_analysis(args.data_dir, args.output_csv)
