# =====================
# Legal Assistant Workflow (Labor Law)
# =====================

# Import system module for environment variables
import os

# Import typing utilities to define the structure of the state passed between steps
from typing import List, TypedDict

# Import the prompt template for building messages to send to the language model
from langchain_core.prompts import ChatPromptTemplate

# Import a fallback language model in case others fail (OpenAI, Gemini, etc.)
from app.llms import fallback_llm

# This will ensure the output from the LLM is returned as a plain string
from langchain_core.output_parsers import StrOutputParser

# Import core LangGraph tools for building step-by-step workflows
from langgraph.graph import StateGraph, START, END

# Import a retriever for laboral law articles (specific to labor law context)
from app.vectorstore.retrievers import laboral_retriever

# Optional: Disable telemetry from Chroma (a vector store backend)
os.environ["CHROMA_TELEMETRY"] = "FALSE"


# =====================
# 1. Prompt Template
# =====================
# This prompt tells the LLM how to behave (as a legal assistant) and how to use retrieved legal articles
prompt = ChatPromptTemplate.from_template(
    """
    Eres un asistente legal.
    Utiliza los artículos recuperados para brindar asistencia legal.
    Proporciona información útil para ayudar al usuario con la pregunta.

    Pregunta: {question}
    Artículos de la ley: {context}
    Respuesta:
    """
)

# Connect the prompt to the language model and string parser as a chain
model = fallback_llm
law_articles_chain = prompt | model | StrOutputParser()


# =====================
# 2. State Definition
# =====================
# This defines the shape of data (or "state") that flows through the graph
class LawArticlesGraphState(TypedDict):
    question: str                          # The user's legal question
    retrieved_law_articles: List[str]     # Articles retrieved by the retriever
    generation: str                        # Final legal answer generated by the LLM


# =====================
# 3. Step 1 - Retrieve Labor Law Articles
# =====================
def retrieve_laboral_articles(state):
    print("---RETRIEVE LABORAL ARTICLES---")
    
    # Extract the question from the incoming state
    question = state["question"]
    
    # Use the laboral retriever to fetch relevant legal articles
    retrieved_law_articles = laboral_retriever.invoke(question)
    
    # Debug print (useful during development)
    print(retrieved_law_articles)
    
    # Return updated state with retrieved articles
    return {"question": question, "retrieved_law_articles": retrieved_law_articles}


# =====================
# 4. Step 2 - Generate Legal Answer
# =====================
def generate_laboral_assistance(state):
    print("---GENERATE LABORAL ASSISTANCE---")
    
    # Extract input values from the state
    question = state["question"]
    retrieved_law_articles = state["retrieved_law_articles"]
    
    # Use the model chain to generate an answer using the question and article context
    generation = law_articles_chain.invoke({
        "question": question,
        "context": retrieved_law_articles
    })
    
    # Return final state including the generated legal assistance
    return {
        "question": question,
        "retrieved_law_articles": retrieved_law_articles,
        "generation": generation,
    }


# =====================
# 5. Define LangGraph Workflow
# =====================
# This function builds the graph by adding nodes and defining the flow of data
def create_laboral_assistant_workflow():
    # Create a new state graph with a specific state structure
    workflow = StateGraph(LawArticlesGraphState)

    # Add both steps (nodes) to the graph
    workflow.add_node("retrieve_laboral_articles", retrieve_laboral_articles)
    workflow.add_node("generate_laboral_assistance", generate_laboral_assistance)

    # Define the edges (flow between steps)
    workflow.add_edge(START, "retrieve_laboral_articles")                       # Start → Retrieve
    workflow.add_edge("retrieve_laboral_articles", "generate_laboral_assistance")  # Retrieve → Generate
    workflow.add_edge("generate_laboral_assistance", END)                      # Generate → End

    # Compile the workflow to make it executable
    return workflow.compile()


# =====================
# 6. Build the Workflow
# =====================
# Create an instance of the compiled LangGraph workflow
laboral_graph = create_laboral_assistant_workflow()
