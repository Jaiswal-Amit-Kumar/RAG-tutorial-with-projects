from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text_loader = TextLoader('python_code.txt', encoding='utf-8')

text_docs = text_loader.load()

text_splitters = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 400,
    chunk_overlap = 0
)

text_results = text_splitters.split_documents(text_docs)
print(text_results[0])

pdf_loader = PyPDFLoader('markdown.pdf')

pdf_docs = pdf_loader.load()

pdf_splitters = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size = 400,
    chunk_overlap = 0
)

pdf_results = pdf_splitters.split_documents(pdf_docs)
print(pdf_results[0])
