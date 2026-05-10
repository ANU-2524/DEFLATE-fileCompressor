"""
DEFLATE File Compressor - Unified Entry Point

This application runs the Streamlit UI which combines:
- Single file compression
- Batch file compression
- Compression history
- Analytics dashboard
- Advanced settings
"""

import subprocess
import sys

def main():
    """Run the Streamlit application"""
    print("=" * 60)
    print("DEFLATE File Compressor - Unified Application")
    print("=" * 60)
    print("\nStarting Streamlit app...")
    print("A browser window will open automatically.\n")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
