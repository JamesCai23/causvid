#!/usr/bin/env python3
"""
Download Wan2.1-T2V-1.3B model from HuggingFace with better timeout handling.
This script handles network issues and can resume interrupted downloads.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse

def download_wan_model(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    local_dir=None,
    use_mirror=False,
    timeout=300.0,
    max_workers=4
):
    """
    Download the Wan model with improved timeout and retry settings.
    
    Args:
        repo_id: HuggingFace repository ID
        local_dir: Local directory to save the model (default: wan_models/Wan2.1-T2V-1.3B/)
        use_mirror: Whether to use HF mirror endpoint
        timeout: Request timeout in seconds (default: 300 for large files)
        max_workers: Number of parallel download workers
    """
    # Set default local directory if not provided
    if local_dir is None:
        script_dir = Path(__file__).parent.absolute()
        local_dir = script_dir / "wan_models" / "Wan2.1-T2V-1.3B"
    else:
        local_dir = Path(local_dir)
    
    # Create directory if it doesn't exist
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {repo_id} to {local_dir}")
    print(f"Timeout: {timeout}s, Workers: {max_workers}")
    
    # Set mirror endpoint if requested
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HuggingFace mirror endpoint: https://hf-mirror.com")
    
    # Configure HTTP timeout for large file downloads
    # This sets the default timeout for requests
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout)
    
    try:
        # Download with improved settings
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,  # Use actual files, not symlinks
            resume_download=True,  # Resume interrupted downloads
            max_workers=max_workers,
            token=None,  # Public repo, no token needed
        )
        
        print(f"\n✓ Successfully downloaded model to: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Try using --use-mirror flag for mirror endpoint")
        print("3. Increase timeout with --timeout flag (e.g., --timeout 600)")
        print("4. Reduce workers with --max-workers flag (e.g., --max-workers 2)")
        print("5. The download will resume from where it left off if you run again")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Wan2.1-T2V-1.3B model from HuggingFace"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="Wan-AI/Wan2.1-T2V-1.3B",
        help="HuggingFace repository ID (default: Wan-AI/Wan2.1-T2V-1.3B)"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Local directory to save the model (default: wan_models/Wan2.1-T2V-1.3B/)"
    )
    parser.add_argument(
        "--use-mirror",
        action="store_true",
        help="Use HuggingFace mirror endpoint (https://hf-mirror.com)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Request timeout in seconds (default: 300)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    download_wan_model(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        use_mirror=args.use_mirror,
        timeout=args.timeout,
        max_workers=args.max_workers
    )
