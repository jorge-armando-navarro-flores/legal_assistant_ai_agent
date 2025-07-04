# ============================
# Legal Assistant FastAPI Server
# ============================

# -------- Standard Library --------
import os

# -------- FastAPI Core Components --------
from fastapi import FastAPI, Depends, Request, Header, HTTPException

# -------- Environment Variable Loader --------
from dotenv import load_dotenv

# -------- Import the Main Legal Agent Workflow --------
from app.agents.legal_assistant_agent import legal_assistant_graph


# ============================
# 1. Load Environment Variables
# ============================

# This loads variables from a `.env` file into the environment
# Useful for storing secrets like API keys
load_dotenv(override=True)


# ============================
# 2. Initialize the FastAPI App
# ============================

# Create an instance of the FastAPI application
app = FastAPI()


# ============================
# 3. Load the API Key
# ============================

# Try to load the API key from the environment
# If not found, use "default-dev-key" (for local development only)
API_KEY = os.getenv("API_KEY", "default-dev-key")


# ============================
# 4. Dependency - API Key Verification
# ============================

# This function checks the x-api-key header in each request
# If the key is wrong or missing, raise a 401 Unauthorized error
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ============================
# 5. Define API Routes
# ============================

# -------- Health Check --------
@app.get("/")
def read_root():
    """
    Simple health check endpoint.
    Useful to verify that the server is running.
    """
    return {"message": "Backend is live."}


# -------- Chatbot Response Endpoint --------
@app.post("/chat/response", dependencies=[Depends(verify_api_key)])
def chat_stream(question: str, request: Request):
    """
    Accepts a legal question and routes it through the legal_assistant_graph.
    Requires a valid API key in the request header.
    """
    # Pass the user’s question to the agent workflow
    response = legal_assistant_graph.invoke({"question": question})

    # Return the structured response (includes category, answer, and docs)
    return response
