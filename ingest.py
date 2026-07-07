import argparse
import sys

import rag


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the Chroma document store.")
    parser.add_argument(
        "folder",
        nargs="?",
        default="./documents",
        help="Folder containing PDF files to ingest (default: ./documents)",
    )
    args = parser.parse_args()

    try:
        collection = rag.get_collection()
    except Exception as e:
        print(f"Error: could not initialize document store: {e}")
        sys.exit(1)

    print(f"Ingesting PDFs from {args.folder} ...")
    try:
        num_files, num_chunks = rag.ingest_pdfs(args.folder, collection)
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)

    if num_files == 0:
        print(f"No PDF files found in {args.folder}.")
    else:
        print(f"Ingested {num_chunks} chunks from {num_files} PDF file(s).")


if __name__ == "__main__":
    main()
