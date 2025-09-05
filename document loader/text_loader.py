from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = "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"

model = ChatOpenAI(model="deepseek/deepseek-r1:free")

prompt = PromptTemplate(
    template='write a summary for the following tittle - \n {title}',
    input_variables=['title']
)

parser = StrOutputParser()

loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs)

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model | parser

final_output = chain.invoke({'title':docs[0].page_content})
print(final_output)