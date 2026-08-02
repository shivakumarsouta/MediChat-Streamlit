# 🩺 MediBot: RAG Based Medical Question Answering Chatbot

MediBot is an intelligent **Retrieval Augmented Generation (RAG)** powered medical chatbot that answers medical questions using a custom PDF knowledge base. Built with **Streamlit**, **LangChain**, **FAISS**, and the **Groq API**, it retrieves the most relevant medical information from indexed documents and generates accurate, context aware responses using high performance open source LLMs.

---

## 🌐 Live Demo

🚀 **Try the application here:**
https://medibot-by-sks.streamlit.app/

---

## ✨ Features

* 📚 RAG pipeline powered by **FAISS** for semantic document retrieval
* 🧠 Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* ⚡ High speed inference through the **Groq API**
* 💬 Interactive Streamlit chat interface
* 📝 Conversation memory for contextual responses
* 📄 Displays retrieved source documents with page numbers and content previews
* 🔍 Accurate retrieval from a custom medical PDF knowledge base

---

## 📂 Project Structure

```text
MediChat-Streamlit/
│
├── app.py                  # Streamlit application
├── connect_llm.py                # LLM connection script
├── create_memory.py              # Builds the FAISS vector database
│
├── vector_db/
│   └── faiss_db/
│       ├── index.faiss
│       └── index.pkl
│
├── data/
│   └── GALE_ENCYCLOPEDIA.pdf      # Medical knowledge base
│
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shivakumarsouta/MediChat-Streamlit.git
cd MediChat-Streamlit
```

### 2. Create a Virtual Environment (Recommended)

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root and add your Groq API key.

```env
GROQ_API_KEY=your_groq_api_key
```

You can obtain an API key from:

https://console.groq.com/

### 5. Build the Vector Database

Run the following command once to generate the FAISS index from the PDF knowledge base.

```bash
python create_memory.py
```

### 6. Launch the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 🧠 How It Works

1. The user submits a medical question through the Streamlit interface.
2. The question is converted into embeddings.
3. FAISS retrieves the most relevant document chunks from the indexed PDF.
4. Retrieved context is combined with the user's question.
5. The prompt is sent to a Groq hosted LLM.
6. The generated answer and supporting document references are displayed.

---

## 🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* FAISS
* Hugging Face Sentence Transformers
* Groq API
* python-dotenv

---

## 📦 Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

Key libraries include:

* streamlit
* langchain
* langchain-community
* langchain-groq
* faiss-cpu
* sentence-transformers
* python-dotenv

---

## 📸 Application Workflow

```text
User Question
      │
      ▼
Generate Embeddings
      │
      ▼
Search FAISS Vector Database
      │
      ▼
Retrieve Relevant Context
      │
      ▼
Groq LLM
      │
      ▼
Medical Response + Source References
```

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 🙏 Acknowledgements

* LangChain
* Streamlit
* Groq
* Hugging Face
* FAISS by Meta AI

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more information.

---

## 👨‍💻 Author

**Shiva Kumar Souta**

* GitHub: https://github.com/shivakumarsouta
* LinkedIn: https://linkedin.com/in/shivakumarsouta
* Portfolio: https://shivakumarsouta-portfolio.vercel.app/
* Email: [shivakumarsouta18@gmail.com](mailto:shivakumarsouta18@gmail.com)

---

⭐ If you found this project helpful, consider giving it a **star** on GitHub.