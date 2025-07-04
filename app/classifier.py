# ================================
# Structured Legal Text Classifier
# ================================

# Import LangChain's prompt template to define custom LLM input instructions
from langchain_core.prompts import ChatPromptTemplate

# Import your default fallback language model (e.g., OpenAI, Gemini, Claude)
from app.llms import fallback_llm

# Pydantic is used to define structured output formats (like JSON schemas)
from pydantic import BaseModel, Field


# ================================
# 1. Load LLM
# ================================

# Assign the default LLM to a variable (could be GPT, Gemini, etc.)
llm = fallback_llm


# ================================
# 2. Create Prompt Template
# ================================

# This prompt instructs the LLM to extract specific structured data
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extrae la información deseada del siguiente texto.

    Solo extrae las propiedades mencionadas en la función 'Classification'.

    Texto:
    {input}
    """
)


# ================================
# 3. Define Expected Output Format (Pydantic Schema)
# ================================

# This class defines the fields we expect the LLM to extract
# These fields will be returned in a structured way (like a typed dictionary)
class Classification(BaseModel):
    category: str = Field(
        description="Categoría legal asignada.",
        enum=["Derecho Laboral", "Derecho Civil", "Derecho Penal", "General"],  # Only these values are allowed
    )
    language: str = Field(
        description="El idioma en el que está escrito el texto.",
        enum=["español", "ingles"],
    )


# ================================
# 4. Bind the Output Schema to the LLM
# ================================

# This wraps the LLM so that it returns output that matches the Classification schema
legal_classifier = llm.with_structured_output(Classification)


# ================================
# 5. Run a Test Input
# ================================

# Sample legal input (user question or text)
inp = "las horas extra se pagan impuestos?"

# First, inject the input into the tagging prompt
prompt = tagging_prompt.invoke({"input": inp})

# Then, run the structured LLM to classify the response
response = legal_classifier.invoke(prompt)


# ================================
# 6. Print the Result
# ================================

# Print the type and value of the extracted category
print(type(response.category))   # Should be <class 'str'>
print(response.category)         # Should be "Derecho Laboral", "Derecho Civil", etc.
