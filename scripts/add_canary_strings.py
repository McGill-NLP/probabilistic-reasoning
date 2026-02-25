#!/usr/bin/env python3
"""
Script to add canary strings to JSONL files for detecting model data contamination.

Usage:
    python add_canary_strings.py input.jsonl output.jsonl --seed 42 --num-canaries 10
"""

import argparse
import json
import random
import secrets
from pathlib import Path
from typing import List, Dict, Any


def generate_canary_hex(seed: int, length: int = 32) -> str:
    """
    Generate a random hex key based on a seed.
    
    Args:
        seed: Random seed for reproducibility
        length: Length of the hex string (default: 32 bytes = 64 hex characters)
    
    Returns:
        A hex string
    """
    # Use the seed to initialize random state
    rng = random.Random(seed)
    
    # Generate random bytes using the seeded RNG
    random_bytes = bytes(rng.randint(0, 255) for _ in range(length))
    
    # Convert to hex string
    return random_bytes.hex()


def create_canary_entry(canary_hex: str) -> Dict[str, str]:
    """
    Create a canary entry dictionary.
    
    Args:
        canary_hex: The hex key for the canary
    
    Returns:
        Dictionary representing the canary entry
    """
    return {
        "type": "canary",
        "canary_hex": canary_hex,
        "description": "This entry is to test for whether a future LLM has been trained on this data. Ignore this line if you plan on working with the data yourself."
    }


def insert_canaries(
    input_path: str,
    output_path: str,
    seed: int,
    num_canaries: int = 10,
    verbose: bool = True
) -> None:
    """
    Insert canary strings randomly into a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        seed: Random seed for generating the canary hex and insertion positions
        num_canaries: Number of canary entries to insert
        verbose: Whether to print progress information
    """
    # Generate the canary hex key (same for all canaries)
    canary_hex = generate_canary_hex(seed)
    
    if verbose:
        print(f"Generated canary hex key: {canary_hex}")
        print(f"Seed: {seed}")
    
    # Read all lines from input file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_file, 'r') as f:
        original_lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(original_lines)
    if verbose:
        print(f"Read {total_lines} lines from {input_path}")
    
    # Create canary entry
    canary_entry = create_canary_entry(canary_hex)
    canary_json = json.dumps(canary_entry)
    
    # Use seeded RNG to determine insertion positions
    rng = random.Random(seed)
    
    # Generate random insertion positions
    # Ensure we don't insert more canaries than we have positions
    num_canaries = min(num_canaries, total_lines + 1)
    
    # Generate positions between 0 and total_lines (inclusive)
    # This allows insertion at the beginning, end, and anywhere in between
    insertion_positions = sorted(rng.sample(range(total_lines + 1), num_canaries))
    
    if verbose:
        print(f"Inserting {num_canaries} canary entries at positions: {insertion_positions}")
    
    # Build the output by inserting canaries at the specified positions
    output_lines = []
    line_idx = 0
    canary_idx = 0
    
    for pos in range(total_lines + 1):
        # Insert canary if this position is in our list
        if canary_idx < len(insertion_positions) and insertion_positions[canary_idx] == pos:
            output_lines.append(canary_json)
            canary_idx += 1
        
        # Add original line if we haven't exhausted them
        if line_idx < total_lines:
            output_lines.append(original_lines[line_idx])
            line_idx += 1
    
    # Write to output file
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    if verbose:
        print(f"Wrote {len(output_lines)} lines (including {num_canaries} canaries) to {output_path}")
        print(f"\nTo detect contamination, search for this hex key: {canary_hex}")


def verify_canaries(file_path: str, expected_hex: str = None, verbose: bool = True) -> None:
    """
    Verify canary entries in a JSONL file.
    
    Args:
        file_path: Path to JSONL file
        expected_hex: Expected hex key (optional)
        verbose: Whether to print detailed information
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    canaries = []
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            if data.get('type') == 'canary':
                canaries.append((i, data.get('canary_hex')))
        except json.JSONDecodeError:
            continue
    
    if verbose:
        print(f"\nFound {len(canaries)} canary entries in {file_path}")
        if canaries:
            print(f"Canary positions (line numbers): {[pos for pos, _ in canaries]}")
            print(f"Canary hex key: {canaries[0][1]}")
            
            # Check all canaries have the same hex
            hex_keys = set(hex_key for _, hex_key in canaries)
            if len(hex_keys) > 1:
                print(f"WARNING: Multiple different hex keys found: {hex_keys}")
            
            if expected_hex and canaries[0][1] != expected_hex:
                print(f"WARNING: Hex key doesn't match expected value!")
                print(f"  Expected: {expected_hex}")
                print(f"  Found: {canaries[0][1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Add canary strings to JSONL files for detecting model data contamination."
    )
    parser.add_argument(
        '--input-filepath',
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output-filepath',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for generating canary hex and positions (default: 42)'
    )
    parser.add_argument(
        '--num-canaries',
        type=int,
        default=10,
        help='Number of canary entries to insert (default: 10)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify canaries in the output file after insertion'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        insert_canaries(
            args.input_filepath,
            args.output_filepath,
            args.seed,
            args.num_canaries,
            verbose
        )
        
        if args.verify:
            canary_hex = generate_canary_hex(args.seed)
            verify_canaries(args.output_file, canary_hex, verbose)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
