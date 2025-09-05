from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser

loader = DirectoryLoader(path = 'books', glob = '*.pdf', loader_cls=PyMuPDFLoader)

docs = loader.load() # for smaller pdfs

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)


docs1 = loader.lazy_load() # for larger pdfs

for documents in docs:
    print(documents.metadata)




