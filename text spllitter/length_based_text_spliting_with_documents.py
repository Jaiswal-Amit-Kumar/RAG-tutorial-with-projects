from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

text_loader = TextLoader('cricket.txt', encoding='utf-8')

text_docs = text_loader.load()

print(text_docs[0].page_content)

text_splitter = CharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap= 0,
    separator= ''
)

text_result = text_splitter.split_documents(text_docs)

print(text_result[0].page_content)

pdf_loader = PyPDFLoader('sample-5-page-pdf-a4-size.pdf')

pdf_docs = pdf_loader.load()

pdf_splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator= ''
)

pdf_result = pdf_splitter.split_documents(pdf_docs)

print(pdf_result[0].page_content)