from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = "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"

model = ChatOpenAI(model="deepseek/deepseek-r1:free")

prompts = PromptTemplate(template='answer the following question \n {question} from  the following text \n {text} '
'and provide all the answers in choosen language only \n {language}',
                         input_variables=['question', 'text', 'language'])

parser = StrOutputParser()

url = 'https://anyweb.ee/en/services/websites/'

loader = WebBaseLoader(url)

docs = loader.load()

# print(type(docs))
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

chain = prompts | model | parser

final_output = chain.invoke({'question': 'какова единственная цель этого сайта?', 'text': docs[0].page_content, 'language':'english'})

print(final_output)
