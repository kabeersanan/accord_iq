# src/utils/chunker.py

from typing import List

def chunk_text(text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
    """
    Splits long text into overlapping chunks.

    Args:
        text (str): The full text to split.
        chunk_size (int): Maximum characters per chunk.
        overlap (int): Overlap between chunks to preserve context.

    Returns:
        List[str]: List of text chunks.
    """
    # Clean text a bit
    text = text.replace("\n", " ").replace("  ", " ").strip()

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        chunks.append(chunk)

        # Move start forward with overlap
        start += chunk_size - overlap

    return chunks
