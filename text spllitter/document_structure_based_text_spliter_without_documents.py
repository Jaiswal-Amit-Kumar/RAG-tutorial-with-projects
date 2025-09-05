from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

python_code_text = '''def split_documents(docs, chunk_size=100, chunk_overlap=0, separator=''):
    """
    Splits a list of documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        docs (List[str]): List of textual documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): Number of characters to overlap between chunks.
        separator (str): Separator used to split text (typically '' or '\n').

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=separator
    )

    # splitter.split_documents expects a list of Document objects in LangChain,
    # but for simplicity, we can directly split raw strings using split_text individually.
    # Let's handle both cases by calling split_text on each document and accumulating the results.

    all_chunks = []
    for doc in docs:
        chunks = splitter.split_text(doc)
        all_chunks.extend(chunks)

    return all_chunks

if __name__ == "__main__":
    # Example documents list
    documents = [
        "This is the first sample document. It contains some text to be split into chunks.",
        "Here is another document, with more content that will be split accordingly."
    ]

    chunks = split_documents(documents)

    print("Split document chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}: {chunk}")
        '''

python_code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 400,
    chunk_overlap = 0

)

python_code_result = python_code_splitter.split_text(python_code_text)

print(len(python_code_result))
print(python_code_result[0])

markdown_text = '''# Document Splitter Using LangChain's RecursiveCharacterTextSplitter

This Python script demonstrates how to split a list of text documents into smaller chunks using LangChain's `RecursiveCharacterTextSplitter`. It is designed to be simple, modular, and production-ready.

---

## Features

- Supports splitting a list of string documents into manageable chunks.
- Customizable chunk size, overlap, and separator parameters.
- Uses LangChain's concrete `RecursiveCharacterTextSplitter` implementation.
- Easy to extend and adapt to different text splitting needs.

---

## Requirements

- Python 3.7+
- LangChain library (`pip install langchain`)

---

## Code Overview


        '''

markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 400,
    chunk_overlap = 0

)

markdown_result = markdown_splitter.split_text(markdown_text)

print(len(markdown_result))
print(markdown_result[0])