from utils.file_handler import FileHandler
from compressor import Compressor
from decompressor import Decompressor
from visualization.visualizer import Visualizer
import json
import os


OUTPUT_DIR = "output"


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def compress_file():
    print("Current Working Directory:", os.getcwd())
    path = input("Enter input file path: ").strip()

    if not os.path.exists(path):
        print("❌ File not found! Please check path.")
        return
    data = FileHandler.read_file(path)

    compressor = Compressor()
    encoded, root, tokens, codes = compressor.compress(data)

    ensure_output_dir()

    # Save compressed data
    FileHandler.write_file(f"{OUTPUT_DIR}/compressed.bin", encoded)

    # Save metadata (IMPORTANT 🔥)
    metadata = {
        "codes": codes,
        "tokens": tokens
    }

    with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("\n✅ Compression Done Successfully!")
    print(f"Compressed file saved at: {OUTPUT_DIR}/compressed.bin")
    print(f"Metadata saved at: {OUTPUT_DIR}/metadata.json")

    # Visualization
    print("\n🌳 Showing Huffman Tree...")
    Visualizer.draw_huffman_tree(root)


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