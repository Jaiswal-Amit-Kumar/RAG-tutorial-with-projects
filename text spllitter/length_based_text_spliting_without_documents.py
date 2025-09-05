from langchain.text_splitter import CharacterTextSplitter

text = '''
Text splitting is the process of dividing a large block of text into smaller, 
more manageable segments or chunks. This technique is commonly used in natural language processing (NLP) 
and text analysis to improve readability, facilitate easier processing, and enable efficient handling of textual data. 
Text can be split based on various criteria such as sentences, paragraphs, phrases, or specific delimiters like 
punctuation marks or line breaks. Effective text splitting helps in tasks like summarization, information retrieval, 
machine translation, and chatbot interactions by breaking down complex information into coherent units that machines 
or humans can better understand and analyze.
'''

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator = ''
)

result = splitter.split_text(text)

print(result)