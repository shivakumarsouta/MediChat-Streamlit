import os

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Load environment token and set Groq model ID
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_ID = "llama-3.3-70b-versatile"

# Initialize the Groq Chat client
client = ChatGroq(model=GROQ_MODEL_ID, temperature=0.1, api_key=GROQ_API_KEY)

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        input_variables=["context", "question"],
        template=custom_prompt_template
    )

# Step 3: Load the vector database
FAISS_DB_PATH = "vector_db/faiss_db"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(FAISS_DB_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Create the custom chain
def run_custom_chain(user_query: str):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    formatted_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=user_query)

    # Invoke Groq chat model
    response = client.invoke(formatted_prompt)

    return {
        "result": response.content,
        "source_documents": docs
    }

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    response = run_custom_chain(user_query)
    print("Answer:", response['result'])

    print("\nSource Documents:\n")
    for i, doc in enumerate(response["source_documents"], start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        print(f"Document {i} — Source: {source}, Page: {page}\nContent: {doc.page_content[:300]}...\n")