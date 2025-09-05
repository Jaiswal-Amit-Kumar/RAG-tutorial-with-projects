from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

doc1 = Document(page_content='Delhi Capitals (DC) is a professional Twenty20 cricket team based in Delhi, competing in the IPL. Owned by GMR Group and JSW Sports, their home ground is Arun Jaitley Stadium. The team is captained by Axar Patel and coached by Hemang Badani. Delhi Capitals reached their first IPL final in 2020 and have a mix of experienced and young players, aiming to be a central force in the IPL landscape.',
                metadata = {'team': 'Delhi Capitals (DC)'})

doc2 = Document(page_content='''Mumbai Indians (MI), based in Mumbai, Maharashtra, is one of the most successful IPL franchises with five IPL titles. Owned by Reliance Industries, they play home matches at Wankhede Stadium. Under the leadership of Hardik Pandya and head coach Mahela Jayawardene, MI's squad features star players like Rohit Sharma and Jasprit Bumrah. The team is known for its balanced lineup and strong bowling attack.''',
                metadata = {'team': 'Mumbai Indians (MI)'})

doc3 = Document(page_content='Chennai Super Kings (CSK) from Chennai, Tamil Nadu, is a highly consistent IPL team with five IPL championships as well. Captained by MS Dhoni and coached by Stephen Fleming, CSK plays at the M. A. Chidambaram Stadium. The franchise boasts a strong record of playoff appearances and is valued as one of the most prestigious IPL teams. They pride themselves on experience and tactical acumen.',
                metadata = {'team': 'Chennai Super Kings (CSK)'})

doc4 = Document(page_content='Kolkata Knight Riders (KKR), based in Kolkata, West Bengal, is jointly owned by Shah Rukh Khan, Juhi Chawla, and Jay Mehta. Known for their iconic purple and gold colors, KKR has won the IPL three times, most recently in 2024. The team plays at Eden Gardens and is currently captained by Ajinkya Rahane. KKR is well-known for a strong team spirit and the motto "Korbo, Lorbo, Jeetbo" (Perform, Fight, Win).',
                metadata = {'team': 'Kolkata Knight Riders (KKR)'})

docs = [doc1, doc2, doc3, doc4]
print(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"trust_remote_code": True}
)
vector_store = FAISS(
    embedding_function = embeddings,
)

# add documents in the vector store
print(vector_store.add_documents(docs))