# Import standard libraries
import os

# Import FastAPI core components and helpers
from fastapi import FastAPI, Depends, Request, Header, HTTPException

# Load environment variables from a .env file
from dotenv import load_dotenv

# Import your chatbot logic from a local module
from app.agents.legal_assistant_agent import legal_assistant_graph

# Load environment variables, overriding existing ones if needed
load_dotenv(override=True)

# Ingest docs
# ingest_docs()

# Initialize the FastAPI app
app = FastAPI()

# Load the API key from environment variables (or use a default for dev)

API_KEY = os.getenv("API_KEY", "default-dev-key")

# ============ DEPENDENCY FUNCTIONS ============

# Dependency to verify API key in request headers
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============ API ENDPOINTS ============


# Basic health check route to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "Backend is live."}


# POST endpoint to stream chatbot responses
@app.post("/chat/response", dependencies=[Depends(verify_api_key)])
def chat_stream(
    question: str, request: Request
):
    # Generate a streaming response from the chatbot
    response = legal_assistant_graph.invoke({"question": question})
    return response

