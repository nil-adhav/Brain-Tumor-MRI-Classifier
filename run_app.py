#!/usr/bin/env python
"""
Script to run the Brain Tumor Classifier Streamlit app
"""
import subprocess
import sys
import os

def run_app():
    """Run the Streamlit app"""
    app_file = os.path.join(os.path.dirname(__file__), "app.py")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_file],
            check=True
        )
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()
