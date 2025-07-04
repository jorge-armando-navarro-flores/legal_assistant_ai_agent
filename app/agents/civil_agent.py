# Importing necessary modules and classes
import os
from typing import List, TypedDict

# LangChain prompt templates
from langchain_core.prompts import ChatPromptTemplate

# Our fallback LLM, in case the primary model fails
from app.llms import fallback_llm

# Parser to ensure the model's output is returned as a string
from langchain_core.output_parsers import StrOutputParser

# LangGraph components for creating a multi-step workflow
from langgraph.graph import StateGraph, START, END

# A retriever that fetches legal documents/articles related to civil law
from app.vectorstore.retrievers import civil_retriever

# Disable telemetry for Chroma (a vector database)
os.environ["CHROMA_TELEMETRY"] = "FALSE"


# 1. Create a Prompt Template for the model
# This prompt helps the model understand how to answer legal questions using retrieved legal articles
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

# 2. Set the LLM (Language Model) pipeline
# Chain the prompt → model → output parser
model = fallback_llm
law_articles_chain = prompt | model | StrOutputParser()


# 3. Define the shape of the state passed between nodes in the workflow
# LangGraph will pass this state dictionary as the input and output of each node
class LawArticlesGraphState(TypedDict):
    question: str
    retrieved_law_articles: List[str]
    generation: str


# 4. Node 1: Retrieve legal articles based on the user's question
def retrieve_laboral_articles(state):
    print("---RETRIEVE LABORAL ARTICLES---")
    question = state["question"]
    
    # Use the retriever to fetch relevant legal documents
    retrieved_law_articles = civil_retriever.invoke(question)
    print(retrieved_law_articles)

    # Return the updated state with the retrieved articles
    return {"question": question, "retrieved_law_articles": retrieved_law_articles}


# 5. Node 2: Generate a legal response based on the retrieved articles
def generate_laboral_assistance(state):
    print("---GENERATE LABORAL ASSISTANCE---")
    question = state["question"]
    retrieved_law_articles = state["retrieved_law_articles"]
    
    # Use the LLM chain to generate a legal explanation or advice
    generation = law_articles_chain.invoke({
        "question": question,
        "context": retrieved_law_articles
    })

    # Return the updated state including the generated response
    return {
        "question": question,
        "retrieved_law_articles": retrieved_law_articles,
        "generation": generation,
    }


# 6. Create the workflow graph using LangGraph
# This defines how nodes are connected and the overall execution order
def create_laboral_assistant_workflow():
    # Initialize a graph that operates on the LawArticlesGraphState
    workflow = StateGraph(LawArticlesGraphState)

    # Add nodes to the graph
    workflow.add_node("retrieve_laboral_articles", retrieve_laboral_articles)
    workflow.add_node("generate_laboral_assistance", generate_laboral_assistance)

    # Define the sequence of operations (edges)
    workflow.add_edge(START, "retrieve_laboral_articles")
    workflow.add_edge("retrieve_laboral_articles", "generate_laboral_assistance")
    workflow.add_edge("generate_laboral_assistance", END)

    # Compile and return the final executable graph
    return workflow.compile()


# 7. Build the workflow
# This object can now be used to process user questions through the defined steps
civil_graph = create_laboral_assistant_workflow()
