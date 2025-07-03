import os
from typing import TypedDict, List
from langgraph.graph import END, START, StateGraph
from app.classifier import legal_classifier, tagging_prompt
from app.agents.labor_agent import laboral_graph
from app.agents.civil_agent import civil_graph
from app.agents.penal_agent import penal_graph
from app.llms import fallback_llm

os.environ["CHROMA_TELEMETRY"] = "FALSE"
# Define the structure of the input state (customer support request)
class LegalRequest(TypedDict):
    question: str
    category: str
    answer: str
    retrieved_docs: List[str]


# Function to categorize the support request
def categorize_request(request: LegalRequest):
    print(f"Received request: {request}")
    # TODO: Implement Conditional Routing
    prompt = tagging_prompt.invoke({"input": request["question"]})
    response = legal_classifier.invoke(prompt)

    if response.category == "Derecho Laboral":
        return "laboral"
    elif response.category == "Derecho Civil":
        return "civil"
    elif response.category == "Derecho Penal":
        return "penal"

    return "fallback"

# Function to process high-priority requests
def handle_laboral(request: LegalRequest):
    print(f"Routing to laboral agent")
    response = laboral_graph.invoke({"question": request["question"]})
    request["category"] = "Derecho Laboral"
    request["retrieved_docs"] = response["retrieved_law_articles"]
    request["answer"] = response["generation"]
    return request

# Function to process standard requests
def handle_civil(request: LegalRequest):
    print(f"Routing to civil agent")
    response = civil_graph.invoke({"question": request["question"]})
    request["category"] = "Derecho Civil"
    request["retrieved_docs"] = response["retrieved_law_articles"]
    request["answer"] = response["generation"]
    return request

# Function to process standard requests
def handle_penal(request: LegalRequest):
    print(f"Routing to penal agent")
    response = penal_graph.invoke({"question": request["question"]})
    request["category"] = "Derecho penal"
    request["retrieved_docs"] = response["retrieved_law_articles"]
    request["answer"] = response["generation"]
    return request

# Function to process standard requests
def handle_fallback(request: LegalRequest) -> str:
    print(f"fallback agent")
    request["category"] = "General"
    request["answer"] = fallback_llm.invoke(
        "No se encontró contexto suficiente. Responde de la mejor manera posible: " + request["question"]
    )
    return request

# Create the state graph
graph = StateGraph(LegalRequest)
# TODO: Create the graph
graph.add_node("laboral",handle_laboral)
graph.add_node("civil",handle_civil)
graph.add_node("penal",handle_penal)
graph.add_node("fallback",handle_fallback)


graph.add_conditional_edges(START,categorize_request)
graph.add_edge("laboral",END)
graph.add_edge("civil",END)
graph.add_edge("penal",END)
graph.add_edge("fallback",END)

legal_assistant_graph = graph.compile()

