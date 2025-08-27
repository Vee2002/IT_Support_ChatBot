# IT_Support_ChatBot

**Overview**

This IT Support Chatbot is an intelligent assistant that helps users solve common problems with their laptops and phones.The bot first searches a curated knowledge base of IT Frequently Asked Questions using semantic similarity search powered by SentenceTransformers.If a relevant answer is not found in the knowledge base, the bot falls back to a Large Language Model to generate a helpful response.

This hybrid approach ensures:

- Fast, consistent answers for common IT issues in phones and laptops.

- Flexible, intelligent responses for questions outside the knowledge base in JSON.

- Real-time interaction via a chat interface

✨**Features**

📂 Knowledge base Q&A: Uses curated IT Frequently Asked Questions (e.g., slow laptop, removal of viruses, overheating laptop).

🧠 Hybrid reasoning: Semantic search for FAQs + LLM fallback for unmatched queries.

⚡ FastAPI backend with WebSocket endpoints.

🖥️ Frontend-ready: Can connect to a simple chat interface.
