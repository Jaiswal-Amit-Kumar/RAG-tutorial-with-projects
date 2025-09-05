from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_loader = TextLoader('cricket.txt', encoding='utf-8')

text_docs = text_loader.load()

print(text_docs)

text_splitter = RecursiveCharacterTextSplitter()

text_result = text_splitter.split_documents(text_docs)

print(text_result[0].page_content)

pdf_loader = PyPDFLoader('sample-5-page-pdf-a4-size.pdf')

pdf_docs = pdf_loader.load()

pdf_splitter = RecursiveCharacterTextSplitter()

pdf_results = pdf_splitter.split_documents(pdf_docs)

print(pdf_results[0].page_content)