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



def decompress_file():
    try:
        encoded = FileHandler.read_file(f"{OUTPUT_DIR}/compressed.bin")

        with open(f"{OUTPUT_DIR}/metadata.json", "r") as f:
            metadata = json.load(f)

        codes = metadata["codes"]
        tokens = metadata["tokens"]

        decompressor = Decompressor()
        decoded_text = decompressor.decompress(encoded, codes, tokens)

        FileHandler.write_file(f"{OUTPUT_DIR}/decompressed.txt", decoded_text)

        print("\n✅ Decompression Done Successfully!")
        print(f"Output saved at: {OUTPUT_DIR}/decompressed.txt")

    except Exception as e:
        print("❌ Error during decompression:", str(e))


def main():
    while True:
        print("\n====== DEFLATE COMPRESSION TOOL ======")
        print("1. Compress File")
        print("2. Decompress File")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == '1':
            compress_file()

        elif choice == '2':
            decompress_file()

        elif choice == '3':
            print("Exiting... 👋")
            break

        else:
            print("❌ Invalid choice, try again.")


if __name__ == "__main__":
    main()