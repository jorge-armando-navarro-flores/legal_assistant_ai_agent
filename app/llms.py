# ============================
# Multi-Provider LLM Fallback Setup
# ============================

# Import LLM wrappers from different providers via LangChain integrations
from langchain_openai import ChatOpenAI                      # OpenAI's GPT models
from langchain_anthropic import ChatAnthropic               # Anthropic's Claude models
from langchain_google_genai import ChatGoogleGenerativeAI   # Google's Gemini models
from langchain_ollama.chat_models import ChatOllama         # Local Ollama-hosted models


# ============================
# 1. Primary LLM: OpenAI GPT-4o Mini
# ============================

# This is the default model the system will try first
openai_llm = ChatOpenAI(model="gpt-4o-mini")


# ============================
# 2. Fallback LLMs
# ============================

# If the primary LLM fails or times out, these will be tried in order

# Claude 3 Opus by Anthropic (high-quality reasoning)
anthropic_llm = ChatAnthropic(model="claude-3-opus-20240229")

# Gemini 2.0 Flash by Google (fast, lightweight)
google_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# LLaMA 3 model via Ollama (runs locally, no API key needed)
ollama_llm = ChatOllama(model="llama3.2:1b")


# ============================
# 3. Combine Models with Fallbacks
# ============================

# `with_fallbacks` allows you to chain models together:
# If one fails, the next one is automatically tried.
fallback_llm = openai_llm.with_fallbacks([
    google_llm,        # Try Gemini if OpenAI fails
    anthropic_llm,     # Try Claude if Gemini fails
    ollama_llm         # Finally, fall back to local LLaMA
])
