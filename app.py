from fastapi import FastAPI,WebSocket,WebSocketDisconnect
# Static Files
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import numpy as np
import os
import json


# Loading environment variables
load_dotenv(".env")

# Zhipu Ai
glm_client = ZhipuAI(api_key=os.getenv("ZHIPU_API"))

# Intializa fast api
app = FastAPI(title='Chatbot Repair with Zhipu AI')

# Static Files
app.mount("/static",StaticFiles(directory="static"),name="static")

# Reading the json file
with open('general_it_faq.json','rb') as f:
    qa_data = json.load(f)

# Converting questions to embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [item["question"] for item in qa_data["questions_answers"]]
answers = [item["answer"] for item in qa_data["questions_answers"]]
question_embeddings = model.encode(questions,convert_to_tensor=True).cpu().numpy()

# Semantic search in our knowledge base
def semantic_search(query,threshold=0.7):
    query_emb = model.encode(query, convert_to_tensor=True).cpu().numpy()
    sims = util.cos_sim(query_emb, question_embeddings)[0].cpu().numpy()
    best_idx = int(np.argmax(sims))
    best_score = sims[best_idx]

    if best_score >= threshold:
        return answers[best_idx]
    return None

# Inference from Zhipu AI
def inference_zhipu(query,raw_answer):
    if raw_answer:
        prompt = f"""
        You are an IT Support Assistant for laptops and phones issues.
        Always respond in a polite, clear, and professional manner.
        Summarize,keep your responses very short and clear.
        The user has asked: "{query}",
        Here's the suggested answer from the knowledge base: {raw_answer}
        Please refine this into a natural response for the user.
        You don't have to give responses in direct speech.
        """
    else:
        prompt = f"""
        You are an IT Support Assistant for laptops and phones issues.
        Always respond in a polite, clear, and professional manner.
        Summarize,keep your responses very short and clear.
        The user asked: "{query}"
        Please provide a helpful IT support response.
        Please refine this into a natural response for the user.
        You don't have to give responses in direct speech.
        """

    response = glm_client.chat.completions.create(
        model="glm-4",
        messages=[
            {"role": "system", "content":"You are a helpful IT support assistant.Keep your answer average in length and ensure they are in English"},
            {"role": "user","content":prompt},
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content


# Request schema
class Query(BaseModel):
    message: str

# Route
@app.post('/chat')
async def chat(query: Query):
    # Step 1: Try semantic search
    raw_answer = semantic_search(query.message)

    # Step 2: Always refine through Zhipu (even if answer exists)
    refined_answer = inference_zhipu(query.message, raw_answer)
    

    return {
        "user_message": query.message,
        "raw_answer": raw_answer if raw_answer else "N/A",
        "final_answer": refined_answer,
    }

# @app.get("/")
# async def root():
#     return {"message": "Chatbot Repair API is running"}

# Chatbot page
@app.get("/",response_class=HTMLResponse)
async def get_chat():
    with open("templates/index.html","r") as f:
        return f.read()
    
# Websocket
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive a message from the frontend
            user_message = await websocket.receive_text()

            # Semantic Search
            raw_answer = semantic_search(user_message)

            # Refine answer with zhipu
            refined_answer = inference_zhipu(user_message,raw_answer)

            # Send response back to frontend
            await websocket.send_json({
                "type": "message",
                "content": refined_answer
            })
    except WebSocketDisconnect:
        print("Client Disconnected")

