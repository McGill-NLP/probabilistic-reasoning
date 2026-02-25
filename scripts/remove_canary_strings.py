#!/usr/bin/env python3
"""
Script to remove canary strings from JSONL files.

This script removes any lines containing {"type": "canary"} entries.
Other lines without a "type" key are preserved.

Usage:
    python remove_canary_strings.py input.jsonl output.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Tuple


def remove_canaries(
    input_path: str,
    output_path: str,
    verbose: bool = True
) -> Tuple[int, int]:
    """
    Remove canary entries from a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file (can be same as input)
        verbose: Whether to print progress information
    
    Returns:
        Tuple of (total_lines_processed, canaries_removed)
    """
    input_file = Path(input_path).resolve()
    output_file = Path(output_path).resolve()
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    canaries_removed = 0
    lines_kept = 0
    total_lines = 0
    
    # If input and output are the same file, read everything into memory first
    same_file = input_file == output_file
    
    if same_file:
        if verbose:
            print(f"Input and output are the same file - reading into memory first")
        with open(input_file, 'r') as f:
            input_lines = [line.strip() for line in f if line.strip()]
    else:
        input_lines = None
    
    # Process the lines
    if same_file:
        # Process from memory
        lines_to_write = []
        for line_num, line in enumerate(input_lines, 1):
            total_lines += 1
            
            try:
                data = json.loads(line)
                
                # Check if this is a canary entry
                if data.get('type') == 'canary':
                    canaries_removed += 1
                    if verbose and canaries_removed <= 5:  # Show first 5 canary positions
                        print(f"Removing canary at line {line_num}")
                    continue
                
                # Keep this line
                lines_to_write.append(line)
                lines_kept += 1
                
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"Warning: Could not parse line {line_num} as JSON: {e}")
                    print(f"  Keeping line as-is")
                # Keep unparseable lines
                lines_to_write.append(line)
                lines_kept += 1
        
        # Write back to the same file
        with open(output_file, 'w') as f_out:
            for line in lines_to_write:
                f_out.write(line + '\n')
    else:
        # Stream from input to output (different files)
        with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                
                total_lines += 1
                
                try:
                    data = json.loads(line)
                    
                    # Check if this is a canary entry
                    if data.get('type') == 'canary':
                        canaries_removed += 1
                        if verbose and canaries_removed <= 5:  # Show first 5 canary positions
                            print(f"Removing canary at line {line_num}")
                        continue
                    
                    # Keep this line
                    f_out.write(line + '\n')
                    lines_kept += 1
                    
                except json.JSONDecodeError as e:
                    if verbose:
                        print(f"Warning: Could not parse line {line_num} as JSON: {e}")
                        print(f"  Keeping line as-is")
                    # Keep unparseable lines
                    f_out.write(line + '\n')
                    lines_kept += 1
    
    if verbose:
        print(f"\nProcessed {total_lines} lines from {input_path}")
        print(f"Removed {canaries_removed} canary entries")
        print(f"Kept {lines_kept} data lines")
        print(f"Output saved to {output_path}")
    
    return total_lines, canaries_removed


def main():
    parser = argparse.ArgumentParser(
        description="Remove canary strings from JSONL files."
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
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        total, removed = remove_canaries(
            args.input_filepath,
            args.output_filepath,
            verbose
        )
        
        if removed == 0 and verbose:
            print("\nNo canary entries found in the input file.")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
