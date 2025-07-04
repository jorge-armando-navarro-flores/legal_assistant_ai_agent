# ============================
# Multi-Agent Legal Assistant Router
# ============================

# Import standard Python libraries
import os
from typing import TypedDict, List

# Import LangGraph tools to define and compile state-based workflows
from langgraph.graph import END, START, StateGraph

# Import the classifier and prompt used to detect legal categories
from app.router import legal_classifier, tagging_prompt

# Import each domain-specific legal agent (graphs for each legal area)
from app.agents.labor_agent import laboral_graph
from app.agents.civil_agent import civil_graph
from app.agents.penal_agent import penal_graph

# Import fallback language model in case routing fails
from app.llms import fallback_llm

# Optional: Disable telemetry reporting from Chroma (the vector store)
os.environ["CHROMA_TELEMETRY"] = "FALSE"


# ============================
# 1. Define the State Structure
# ============================

# This is the format of data passed between nodes in the graph
class LegalRequest(TypedDict):
    question: str              # User's legal question
    category: str              # Detected legal category (laboral, penal, civil)
    answer: str                # Final response to the user
    retrieved_docs: List[str]  # List of retrieved legal documents/articles


# ============================
# 2. Step 1 - Classify the Request
# ============================

# This node will run first: it determines the category of the user's question
def categorize_request(request: LegalRequest):
    print(f"Received request: {request}")

    # Generate a taggable prompt based on the question
    prompt = tagging_prompt.invoke({"input": request["question"]})

    # Use a classifier to label the legal category (laboral, civil, penal)
    response = legal_classifier.invoke(prompt)

    # Route to the correct agent based on classification
    if response.category == "Derecho Laboral":
        return "laboral"
    elif response.category == "Derecho Civil":
        return "civil"
    elif response.category == "Derecho Penal":
        return "penal"

    # If no category matches, fallback to generic response
    return "fallback"


# ============================
# 3. Step 2 - Domain-Specific Handlers
# ============================

# Each of the following functions calls a domain-specific LangGraph agent
# Then, updates the state with retrieved docs and final answer

def handle_laboral(request: LegalRequest):
    print(f"Routing to laboral agent")
    response = laboral_graph.invoke({"question": request["question"]})
    request["category"] = "Derecho Laboral"
    request["retrieved_docs"] = response["retrieved_law_articles"]
    request["answer"] = response["generation"]
    return request


def handle_civil(request: LegalRequest):
    print(f"Routing to civil agent")
    response = civil_graph.invoke({"question": request["question"]})
    request["category"] = "Derecho Civil"
    request["retrieved_docs"] = response["retrieved_law_articles"]
    request["answer"] = response["generation"]
    return request


def handle_penal(request: LegalRequest):
    print(f"Routing to penal agent")
    response = penal_graph.invoke({"question": request["question"]})
    request["category"] = "Derecho penal"
    request["retrieved_docs"] = response["retrieved_law_articles"]
    request["answer"] = response["generation"]
    return request


# ============================
# 4. Step 3 - Fallback Handler
# ============================

# If the classifier fails or returns an unknown category, use a generic response
def handle_fallback(request: LegalRequest) -> LegalRequest:
    print(f"fallback agent")
    request["category"] = "General"
    request["answer"] = fallback_llm.invoke(
        "No se encontró contexto suficiente. Responde de la mejor manera posible: "
        + request["question"]
    )
    return request


# ============================
# 5. Build the LangGraph Workflow
# ============================

# Initialize the workflow with the state shape (LegalRequest)
graph = StateGraph(LegalRequest)

# Register the processing nodes (steps)
graph.add_node("laboral", handle_laboral)
graph.add_node("civil", handle_civil)
graph.add_node("penal", handle_penal)
graph.add_node("fallback", handle_fallback)

# Add the conditional router that runs at START and decides the next step
graph.add_conditional_edges(START, categorize_request)

# Define how each node leads to the END of the process
graph.add_edge("laboral", END)
graph.add_edge("civil", END)
graph.add_edge("penal", END)
graph.add_edge("fallback", END)

# Compile the entire routing system into an executable graph
legal_assistant_graph = graph.compile()
