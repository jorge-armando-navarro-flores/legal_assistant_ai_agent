from langchain_core.prompts import ChatPromptTemplate
from app.llms import fallback_llm
from pydantic import BaseModel, Field


llm = fallback_llm

tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extrae la información deseada del siguiente texto.

    Solo extrae las propiedades mencionadas en la función 'Classification'.

    Texto:
    {input}
    """
    )


class Classification(BaseModel):
    category: str = Field(
        description=(
            "Categoría legal asignada."
        ),
        enum=["Derecho Laboral", "Derecho Civil", "Derecho Penal", "General"]
    )
    language: str = Field(description="El idioma en el que está escrito el texto.", enum=["español", "ingles"])


# Structured LLM
legal_classifier = llm.with_structured_output(Classification)

inp = "las horas extra se pagan impuestos?"
prompt = tagging_prompt.invoke({"input": inp})
response = legal_classifier.invoke(prompt)
print(type(response.category))
print(response.category)