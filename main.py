import os

from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import OpenAI

load_dotenv()

if __name__ == "__main__":
    print("Hello PDF Chatter")

    #loader
    loader = PyPDFLoader(file_path="ReAct.pdf")
    documents = loader.load()
    #print(len(documents))
    #Text Splitter to control character
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30,separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    #embeddings
    embeddings = OpenAIEmbeddings()

    #vector store
    faiss_vectorstore = FAISS.from_documents(documents=docs,embedding=embeddings)

    #store the index locally --> folder name
    #faiss_vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react",embeddings)

    #QA chain --> newer version of VectorDBQA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(),chain_type="stuff",retriever=new_vectorstore.as_retriever())

    res = qa.run("Give me the gist of ReAct in 3 sentences")
  
    print(res) 