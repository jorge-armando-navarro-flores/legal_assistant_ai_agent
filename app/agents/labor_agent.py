import os
from typing import List, TypedDict
from langchain_core.prompts import ChatPromptTemplate
from app.llms import fallback_llm
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from app.vectorstore.retriever import laboral_retriever

os.environ["CHROMA_TELEMETRY"] = "FALSE"


# Prompt for Current Affairs News Summarization
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
model = fallback_llm
law_articles_chain = prompt | model | StrOutputParser()


class LawArticlesGraphState(TypedDict):
    question: str
    retrieved_law_articles: List[str]
    generation: str


# TODO: Use the retriever and retrieve the matching news
def retrieve_laboral_articles(state):
    print("---RETRIEVE LABORAL ARTICLES---")
    question = state["question"]
    retrieved_law_articles = laboral_retriever.invoke(question)
    print(retrieved_law_articles)
    return {"question": question, "retrieved_law_articles": retrieved_law_articles}


# TODO: Summarize the news
# News Summary Generation Node
def generate_laboral_assistance(state):
    print("---GENERATE LABORAL ASSISTANCE---")
    question = state["question"]
    retrieved_law_articles = state["retrieved_law_articles"]
    generation = law_articles_chain.invoke(
        {"question": question, "context": retrieved_law_articles}
    )
    return {
        "question": question,
        "retrieved_law_articles": retrieved_law_articles,
        "generation": generation,
    }


# Current Affairs News Workflow Definition
def create_laboral_assistant_workflow():
    workflow = StateGraph(LawArticlesGraphState)
    workflow.add_node("retrieve_laboral_articles", retrieve_laboral_articles)
    workflow.add_node("generate_laboral_assistance", generate_laboral_assistance)
    workflow.add_edge(START, "retrieve_laboral_articles")
    workflow.add_edge("retrieve_laboral_articles", "generate_laboral_assistance")
    workflow.add_edge("generate_laboral_assistance", END)
    return workflow.compile()


# Execute the Current Affairs News Workflow
laboral_graph = create_laboral_assistant_workflow()
