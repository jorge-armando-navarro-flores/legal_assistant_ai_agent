from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama


openai_llm = ChatOpenAI(model="gpt-4o-mini")
anthropic_llm = ChatAnthropic(model="claude-3-opus-20240229")
google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
ollama_llm = ChatOllama(model="llama3.2:1b")

fallback_llm = openai_llm.with_fallbacks([google_llm, anthropic_llm, ollama_llm])
