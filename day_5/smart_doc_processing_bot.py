from pathlib import Path
import re

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter
)


# ----------------------------
# Detect Document Type
# ----------------------------
def detect_document_type(file_path):

    extension = Path(file_path).suffix.lower()

    if extension == ".md":
        return "markdown"

    elif extension == ".py":
        return "python"

    elif extension == ".txt":
        return "text"

    else:
        return "unsupported"


# ----------------------------
# Recommend Settings
# ----------------------------
def recommend_settings(doc_type):

    recommendations = {
        "markdown": {
            "splitter": "MarkdownHeaderTextSplitter",
            "overlap": 75
        },
        "python": {
            "splitter": "PythonCodeTextSplitter",
            "overlap": 100
        },
        "text": {
            "splitter": "RecursiveCharacterTextSplitter",
            "overlap": 50
        }
    }

    return recommendations.get(doc_type)


# ----------------------------
# Load Document
# ----------------------------
def load_document(file_path):

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    except FileNotFoundError:
        print(" File not found.")
        return None

    except Exception as e:
        print(" Error:", e)
        return None


# ----------------------------
# Choose Splitter
# ----------------------------
def get_splitter(doc_type, overlap, splitter_name=None):

    if splitter_name is None:
        splitter_name = recommend_settings(doc_type)["splitter"]

    if splitter_name == "MarkdownHeaderTextSplitter":

        headers = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]

        return MarkdownHeaderTextSplitter(
            headers_to_split_on=headers
        )

    elif splitter_name == "PythonCodeTextSplitter":

        return PythonCodeTextSplitter(
            chunk_size=500,
            chunk_overlap=overlap
        )

    else:

        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=overlap
        )


# ----------------------------
# Token Counter
# ----------------------------
def count_tokens(text):

    return len(text.split())


# ----------------------------
# Extract Python Functions
# ----------------------------
def extract_function_names(text):

    return re.findall(
        r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        text
    )


# ----------------------------
# Process Document
# ----------------------------
def process_document(content,
                     file_path,
                     doc_type,
                     splitter):

    chunks_output = []

    if doc_type == "markdown":

        chunks = splitter.split_text(content)

        for chunk in chunks:

            metadata = {
                "source": file_path,
                "type": doc_type,
                "section": chunk.metadata,
                "tokens": count_tokens(
                    chunk.page_content
                )
            }

            chunks_output.append({
                "text": chunk.page_content,
                "metadata": metadata
            })

    else:

        chunks = splitter.split_text(content)

        for chunk in chunks:

            metadata = {
                "source": file_path,
                "type": doc_type,
                "tokens": count_tokens(chunk)
            }

            if doc_type == "python":

                metadata["functions"] = (
                    extract_function_names(chunk)
                )

            chunks_output.append({
                "text": chunk,
                "metadata": metadata
            })

    return chunks_output


# ----------------------------
# Display Results
# ----------------------------
def display_results(chunks):

    print("\n Processing Complete")
    print("Total Chunks:", len(chunks))

    preview_count = min(3, len(chunks))

    for i in range(preview_count):

        chunk = chunks[i]

        print("\n" + "=" * 60)
        print(f"Chunk {i + 1}")

        print("\nMetadata:")

        for key, value in chunk["metadata"].items():
            print(f"{key}: {value}")

        print("\nPreview:")
        print(chunk["text"][:300])

        print("=" * 60)


# ----------------------------
# Main Interactive Loop
# ----------------------------
def main():

    print("=" * 60)
    print("SMART DOCUMENT PROCESSOR")
    print("=" * 60)

    while True:

        file_path = input(
            "\nEnter file path (i.e .txt , .md ,.py): "
        ).strip().strip('"')

        if not Path(file_path).exists():

            print(" File does not exist.")
            continue

        doc_type = detect_document_type(file_path)

        if doc_type == "unsupported":

            print(" Unsupported file type.")
            continue

        recommendation = recommend_settings(
            doc_type
        )

        print("\nDetected File Type:",
              doc_type)

        print("Recommended Splitter:",
              recommendation["splitter"])

        print("Recommended Overlap:",
              recommendation["overlap"])

        splitter_name = recommendation[
            "splitter"
        ]

        overlap = recommendation[
            "overlap"
        ]

        choice = input(
            "\nUse recommended settings? (y/n): "
        ).lower()

        if choice == "n":

            print("\nChoose Splitter")
            print(
                "1. RecursiveCharacterTextSplitter"
            )
            print(
                "2. MarkdownHeaderTextSplitter"
            )
            print(
                "3. PythonCodeTextSplitter"
            )

            option = input(
                "Enter choice: "
            )

            mapping = {
                "1":
                "RecursiveCharacterTextSplitter",

                "2":
                "MarkdownHeaderTextSplitter",

                "3":
                "PythonCodeTextSplitter"
            }

            splitter_name = mapping.get(
                option,
                splitter_name
            )

            try:

                overlap = int(
                    input(
                        "Enter overlap value: "
                    )
                )

            except ValueError:

                print(
                    "Invalid overlap."
                    " Using recommended value."
                )

        content = load_document(file_path)

        if content is None:
            continue

        print(
            "\n Document Loaded Successfully"
        )

        print(
            "Document Length:",
            len(content),
            "characters"
        )

        splitter = get_splitter(
            doc_type,
            overlap,
            splitter_name
        )

        chunks = process_document(
            content,
            file_path,
            doc_type,
            splitter
        )

        display_results(chunks)

        again = input(
            "\nProcess another document? (y/n): "
        ).lower()

        if again != "y":

            print(
                "\nThank you for using "
                "Smart Document Processor!"
            )

            break


# ----------------------------
# Run Application
# ----------------------------
if __name__ == "__main__":
    main()