# =============================
# Penal Law Assistant Workflow
# =============================

# Import standard and typing modules
import os
from typing import List, TypedDict

# Import prompt tools to create structured messages for LLM
from langchain_core.prompts import ChatPromptTemplate

# Import the fallback language model (e.g., OpenAI, Gemini, etc.)
from app.llms import fallback_llm

# Ensures LLM output is returned as a simple string
from langchain_core.output_parsers import StrOutputParser

# LangGraph utilities to create and manage step-by-step workflows
from langgraph.graph import StateGraph, START, END

# Import the retriever for penal law documents
from app.vectorstore.retrievers import penal_retriever

# Optional: disable Chroma telemetry if you're using it as a vector DB backend
os.environ["CHROMA_TELEMETRY"] = "FALSE"


# =============================
# 1. Define Prompt Template
# =============================

# This prompt guides the LLM to act like a legal assistant
# It will use retrieved articles to respond to legal questions
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

# Set up the model pipeline: prompt → LLM → string output
model = fallback_llm
law_articles_chain = prompt | model | StrOutputParser()


# =============================
# 2. Define Graph State Format
# =============================

# This structure defines what data is passed between each step of the graph
class LawArticlesGraphState(TypedDict):
    question: str                      # User's legal question
    retrieved_law_articles: List[str] # Articles found by the retriever
    generation: str                   # Final legal response from the LLM


# =============================
# 3. Node 1 - Retrieve Penal Articles
# =============================

# Retrieves relevant law articles for a penal law question
def retrieve_laboral_articles(state):
    print("---RETRIEVE LABORAL ARTICLES---")
    
    # Extract the user question
    question = state["question"]
    
    # Call the penal retriever to get legal documents
    retrieved_law_articles = penal_retriever.invoke(question)
    
    # Print the results (useful for debugging)
    print(retrieved_law_articles)
    
    # Return updated state with retrieved documents
    return {"question": question, "retrieved_law_articles": retrieved_law_articles}


# =============================
# 4. Node 2 - Generate Penal Law Answer
# =============================

# Uses the LLM to generate a helpful legal response using retrieved context
def generate_laboral_assistance(state):
    print("---GENERATE LABORAL ASSISTANCE---")
    
    # Extract question and retrieved context
    question = state["question"]
    retrieved_law_articles = state["retrieved_law_articles"]
    
    # Run the model pipeline to get a response
    generation = law_articles_chain.invoke({
        "question": question,
        "context": retrieved_law_articles
    })

    # Return final state including the answer
    return {
        "question": question,
        "retrieved_law_articles": retrieved_law_articles,
        "generation": generation,
    }


# =============================
# 5. Define LangGraph Workflow
# =============================

# This function builds and compiles the penal law agent graph
def create_laboral_assistant_workflow():
    # Define the graph structure using the state format
    workflow = StateGraph(LawArticlesGraphState)

    # Add both processing nodes to the graph
    workflow.add_node("retrieve_laboral_articles", retrieve_laboral_articles)
    workflow.add_node("generate_laboral_assistance", generate_laboral_assistance)

    # Connect nodes in order: START → retrieve → generate → END
    workflow.add_edge(START, "retrieve_laboral_articles")
    workflow.add_edge("retrieve_laboral_articles", "generate_laboral_assistance")
    workflow.add_edge("generate_laboral_assistance", END)

    # Compile the graph so it can be executed
    return workflow.compile()


# =============================
# 6. Compile the Penal Law Agent
# =============================

# This variable holds the executable LangGraph for penal law questions
penal_graph = create_laboral_assistant_workflow()
