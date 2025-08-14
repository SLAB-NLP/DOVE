#!/usr/bin/env python3
"""
HuggingFace Authentication Module
Handles HuggingFace authentication for all analysis scripts
"""

import os

from dotenv import load_dotenv
from huggingface_hub import login


# Load environment variables from .env file
load_dotenv()
# Global authentication state - ensures one-time authentication only
_hf_authenticated = False


def ensure_hf_authentication():
    """
    Ensure HuggingFace authentication happens exactly once per process.
    Uses lazy loading to avoid multiple login attempts.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    global _hf_authenticated

    if _hf_authenticated:
        return True

    # Read token from environment (loaded via dotenv)
    hf_token = os.getenv('HF_ACCESS_TOKEN')

    # Skip if token is None or "None" string
    if not hf_token or hf_token == "None":
        hf_token = None

    if hf_token:
        try:
            login(token=hf_token)
            print("✅ HuggingFace authentication successful")
            _hf_authenticated = True
            return True
        except Exception as e:
            error_msg = str(e)
            if "Invalid user token" in error_msg or "401" in error_msg or "403" in error_msg:
                print(f"❌ HuggingFace authentication failed: {error_msg}")
                print("💡 Please run: huggingface-cli login")
                print("   Or update your HF_ACCESS_TOKEN environment variable")
                exit(1)
            elif "Too Many Requests" in error_msg or "429" in error_msg:
                print(f"⚠️  HuggingFace rate limit exceeded: {error_msg}")
                print("💡 Please wait a few minutes and try again")
                exit(1)
            else:
                print(f"⚠️  HuggingFace authentication warning: {error_msg}")
                return False
    else:
        print("❌ No HF_ACCESS_TOKEN found in environment variables")
        print("💡 Please set HF_ACCESS_TOKEN in .env file or run: huggingface-cli login")
        exit(1)
