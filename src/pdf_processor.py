from typing import List
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdfs(pdf_files: List[str]) -> str:
    """
    Extracts raw text content from a list of PDF files.

    Iterates through each provided PDF path, opens the document using PyMuPDF (fitz),
    and extracts text from each page. Text from all pages and all documents
    is concatenated into a single string, with newline characters added
    between pages for basic structure.

    Args:
        pdf_files (List[str]): A list of file paths to the PDF documents.

    Returns:
        str: A single string containing all extracted text from the PDFs.
             Returns an empty string if no files are provided or if text
             extraction fails for all provided files.

    Raises:
        Prints an error message to stderr if a specific PDF file cannot be
        opened or read, but continues processing other files.
    """
    full_text = ""
    # Process each PDF file provided in the list.
    for pdf_path in pdf_files:
        try:
            # Open the PDF document using PyMuPDF.
            doc = fitz.open(pdf_path)
            # Iterate through each page of the document.
            for page in doc:
                # Extract text from the current page and append it+ Add a newline character to separate content from different pages.
                full_text += page.get_text() + "\n"
            doc.close() # Close the document.
        except Exception as e:
            # in log error the function will continue attempting to process remaining files.
            print(f"Error reading PDF file {pdf_path}: {e}")
            # Consider more robust error handling for production:
            # - Log to a dedicated logging system.
            # - Raise a custom exception if failure is critical.
            # - Return partial results with error indicators.
    return full_text

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[str]:
    """
    Splits a long text document into smaller, potentially overlapping chunks.

    Uses LangChain's RecursiveCharacterTextSplitter, which tries to split text
    based on common separators (like newlines, spaces) recursively to keep
    semantically related pieces of text together within chunks.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The maximum desired size of each chunk
            (measured by character length). Defaults to 1000.
        chunk_overlap (int, optional): The number of characters to overlap
            between consecutive chunks. This helps maintain context across chunk
            boundaries. Defaults to 150.

    Returns:
        List[str]: A list containing the text chunks. Returns an empty list if
                   the input text is empty or None.
    """
    # Handle empty input text gracefully.
    if not text:
        return []

    # Initialize the text splitter.
    # RecursiveCharacterTextSplitter is generally recommended for arbitrary text.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,          # Max characters per chunk.
        chunk_overlap=chunk_overlap,    # Characters shared between adjacent chunks.
        length_function=len,            # Use standard character length measurement.
        add_start_index=True,           # Include the start index of the chunk in metadata (useful for referencing source).
    )

    # Split the text into LangChain Document objects.
    # create_documents expects a list of texts; here we process one large text.
    documents = text_splitter.create_documents([text])

    # Extract the actual text content from each Document object.
    chunks = [doc.page_content for doc in documents]

    # Informational print statement (useful for debugging/logging).
    print(f"Text split into {len(chunks)} chunks.")
    return chunks