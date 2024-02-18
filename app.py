import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
import os 


with st.sidebar:
    st.title("LLM")

load_dotenv()
def main():
    st.header('chat with pdf')
    #st.write('hi')

    pdf=st.file_uploader('upload the damn pdf',type='pdf')

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        #st.write(pdf_reader)

        text=''
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        #st.write(text)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks)

        #embeddings
        #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        store_name=pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            #st.write('embedding loaded from the disk')
            
        else:
            embeddings = HuggingFaceEmbeddings()
            VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)


        #accept input
        query = st.text_input("ask qn abt the pdf")
        #st.write(query)

        if query:
            docs=VectorStore.similarity_search(query=query,k=3)
            #st.write(docs)
            llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)



if __name__=='__main__':    
    main()