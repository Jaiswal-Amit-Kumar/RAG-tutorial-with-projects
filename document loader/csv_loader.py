from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('industry.csv')

docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)