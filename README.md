🤖 Government Schemes RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot that helps users discover and understand Indian government schemes. The chatbot combines semantic search using FAISS vector database with Google's Gemini AI to provide accurate, context-aware responses about various government welfare programs.
📋 Table of Contents

Overview
Features
How It Works
Technology Stack
Installation
Configuration
Usage
Project Structure
Supported Schemes
Examples
Contributing
License

🎯 Overview
This RAG-powered chatbot helps users:

Discover government schemes based on their needs
Understand eligibility criteria for various programs
Learn about benefits and required documents
Get personalized responses using conversational AI

The system uses vector embeddings to find relevant schemes from a knowledge base and Gemini AI to generate human-like responses, ensuring accurate and contextual answers.
✨ Features

Semantic Search: Uses sentence transformers to understand query intent, not just keywords
Vector Database: FAISS-based efficient similarity search for quick retrieval
RAG Architecture: Combines retrieved context with generative AI for accurate responses
Conversation Memory: Maintains chat history for contextual follow-up questions
Flexible Queries: Handles both scheme-specific and general conversation
Persistent Storage: Saves chat history and vector indices for session continuity

🔍 How It Works

Embedding Generation: All scheme information is converted into vector embeddings using sentence-transformers
Vector Storage: Embeddings are stored in a FAISS index for fast similarity search
Query Processing: User queries are embedded and compared against stored vectors
Context Retrieval: Top-K most relevant schemes are retrieved based on similarity scores
Response Generation: Retrieved context is fed to Gemini AI to generate natural responses
Memory Management: Chat history is maintained for contextual conversations

🛠 Technology Stack

Python 3.8+
FAISS: Facebook AI Similarity Search for vector operations
Sentence Transformers: For generating semantic embeddings (all-MiniLM-L6-v2 model)
Google Gemini AI: For natural language generation (gemini-2.5-flash)
Pandas & NumPy: Data manipulation and processing
JSON: Chat history and metadata storage

📦 Installation
Prerequisites

Python 3.8 or higher
pip package manager
Google Gemini API key

Step-by-Step Setup

Clone the repository

bashgit clone https://github.com/yourusername/govt-schemes-rag-chatbot.git
cd govt-schemes-rag-chatbot

Install dependencies

bashpip install -r requirements.txt

Prepare the dataset
Ensure you have schemes_dataset.csv in the project root with columns:

scheme_name
objective
benefit
eligibility
documents_required


Configure API key
Replace the API key in schm.py:

python   GOOGLE_API_KEY = "your-google-gemini-api-key"
Or set as environment variable:
bash   export GOOGLE_API_KEY="your-google-gemini-api-key"

Run the chatbot

bashpython schm.py
⚙️ Configuration
Key configuration variables in schm.py:
pythonDATASET_FILE = "schemes_dataset.csv"          # Schemes data source
CHAT_HISTORY_FILE = "chat_history.json"       # Conversation storage
FAISS_INDEX_FILE = "schemes_faiss.index"      # Vector index
METADATA_FILE = "schemes_metadata.json"       # Scheme metadata
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"        # Embedding model
TOP_K = 5                                     # Number of schemes to retrieve
💻 Usage
Interactive Mode
Start the chatbot:
bashpython schm.py
Example interactions:
You: hi chatbot
Bot: Hi there! How can I assist you today?

You: what schemes are available for farmers?
Bot: There's a great scheme called PM-Kisan Samman Nidhi...

You: what documents do I need?
Bot: For PM-Kisan, you'll need: Aadhaar Card, Bank Account Details...
Programmatic Usage
pythonfrom schm import answer_query

# Ask a question
response = answer_query("Tell me about education schemes")
print(response)
📁 Project Structure
govt-schemes-rag-chatbot/
│
├── schm.py                      # Main chatbot application
├── requirements.txt             # Python dependencies
├── schemes_dataset.csv          # Government schemes data
├── schemes_faiss.index          # FAISS vector index (generated)
├── schemes_metadata.json        # Scheme metadata (generated)
├── chat_history.json           # Conversation history (generated)
├── ques.txt                    # Q&A documentation
└── README.md                   # This file
🏛 Supported Schemes
Currently includes information about:

PM-Kisan Samman Nidhi (PM-KISAN)

Income support for farmers (₹6,000/year)


Pradhan Mantri Awas Yojana (PMAY)

Affordable housing with interest subsidy


Ayushman Bharat Yojana (PM-JAY)

Health coverage up to ₹5 lakhs/family


PM Ujjwala Yojana

Free LPG connections for BPL households


National Scholarship Portal (NSP)

Financial assistance for students (₹10,000-50,000)



📝 Examples
Query about eligibility:
User: Who can apply for Ayushman Bharat?
Bot: Families identified by SECC 2011 and workers in unorganized 
     sector are eligible. It provides free hospitalization up to 
     ₹5 lakhs per family per year.
Query about documents:
User: What documents needed for PM Awas Yojana?
Bot: For PMAY, you'll need: Aadhaar Card, Income Proof, and 
     Property Papers to avail housing subsidy.
General conversation:
User: What is my name?
Bot: Based on our chat history, your name is Shreya!
🤝 Contributing
Contributions are welcome! Here's how you can help:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

Areas for Improvement

Add more government schemes to the database
Implement multi-language support (Hindi, regional languages)
Add a web interface using Streamlit or Flask
Enhance conversation context with better memory management
Add scheme comparison features
Implement application status tracking

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

Sentence Transformers for semantic embedding models
FAISS by Facebook AI Research for efficient similarity search
Google Gemini for natural language generation
Government of India for scheme information


Note: This is an educational project. Always verify scheme details from official government sources before applying.
Made with ❤️ for helping citizens discover government schemes
