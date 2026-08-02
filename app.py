import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

@st.cache_resource
def load_vector_db(faiss_db_path):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template
    )

def main():
    st.set_page_config(page_title="MediBot", page_icon="Images/favicon.ico", layout="wide")
    st.title("Ask MediBot!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Enter your question here:")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        custom_prompt_template = """
            Use the pieces of information provided in the context to answer user's question.
            If you dont know the answer, just say that you dont know, dont try to make up an answer. 
            Dont provide anything out of the given context

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
        """
        prompt_template = set_custom_prompt(custom_prompt_template)
        
        groq_api_key = os.environ.get("GROQ_API_KEY")
        groq_model_id = "llama-3.3-70b-versatile"

        try:
            vector_store = load_vector_db("vector_db/faiss_db")
            if vector_store is None:
                st.error("Failed to load the vector database.")

            client = ChatGroq(model=groq_model_id, temperature=0.1, api_key=groq_api_key)
            
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)
            context = "\n\n".join(doc.page_content for doc in docs)
            
            formatted_prompt = prompt_template.format(context=context, question=prompt)

            response = client.invoke(formatted_prompt)
            response_content = response.content

            st.chat_message("assistant").markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content})

            with st.expander("Show source documents"):
                for i, doc in enumerate(docs, start=1):
                    source = doc.metadata.get("source", "unknown")
                    page = doc.metadata.get("page", "unknown")
                    st.markdown(f"**Document {i}** — Source: `{source}`, Page: `{page}`\n\n> {doc.page_content[:300]}...")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "An error occurred while processing your request."})

if __name__ == "__main__":
    main()