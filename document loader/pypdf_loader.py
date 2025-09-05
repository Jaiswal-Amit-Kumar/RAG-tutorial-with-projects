from langchain_community.document_loaders import PyPDFLoader
from  langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

loader = PyPDFLoader(r'C:\Users\gauta\Downloads\RAG tutorial\document loader\books\sample-5-page-pdf-a4-size.pdf')

docs = loader.load()

print(docs)
print(type(docs))
print(len(docs))
print(docs[1].page_content)
print(docs[1].metadata)