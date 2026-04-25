import logging
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ── Version-compatible imports ────────────────────────────────────────────────
# langchain 0.2.x keeps chains/memory in `langchain.*`
# langchain 1.x+ moved them to `langchain_classic.*`
try:
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
except ImportError:
    from langchain_classic.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferWindowMemory
    from langchain_classic.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.config import settings
from src.generation.prompt import get_rag_chat_prompt, get_condense_question_prompt

logger = logging.getLogger(__name__)

# LLM loader 

def get_llm(streaming: bool = False):
    callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
    
    # Prioritize Groq if key is present
    if settings.GROQ_API_KEY:
        return ChatGroq(
            model_name=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            groq_api_key=settings.GROQ_API_KEY,
            streaming=streaming,
            callbacks=callbacks
        )

    # Configure OpenRouter if key is present
    if settings.OPENROUTER_API_KEY:
        return ChatOpenAI(
            model_name=settings.LLM_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            streaming=streaming,
            callbacks=callbacks,
            default_headers={
                "HTTP-Referer": "https://github.com/antigravity-ai/rag-project", # Optional
                "X-Title": "RAG Project"
            }
        )
    
    # Fallback to standard OpenAI
    return ChatOpenAI(
        model_name=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        api_key=settings.OPENAI_API_KEY,
        streaming=streaming,
        callbacks=callbacks
    )


# LCEL RAG chain
def build_rag_chain(retriever: BaseRetriever):
    prompt = get_rag_chat_prompt()
    llm    = get_llm()

    def format_context(docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context":  retriever | format_context,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# Conversational RAG chain 

def build_conversational_chain(retriever: BaseRetriever) -> ConversationalRetrievalChain:
    llm    = get_llm()
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=get_condense_question_prompt(),
        return_source_documents=True,    # returns chunks used — for citations
        verbose=False
    )

    return chain


# Single turn QA (simple wrapper) 
def build_qa_chain(retriever: BaseRetriever) -> RetrievalQA:
    llm = get_llm()

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",              # stuff = paste all chunks into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": get_rag_chat_prompt()
        }
    )

    return chain

# Answer + citations
def query_with_citations(chain, question: str) -> dict:
    # Handle dict-based chain call (RetrievalQA)
    result = chain.invoke({"query": question})

    answer = result.get("result", "")
    source_docs = result.get("source_documents", [])

    sources = [
        {
            "source":      doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "?"),
            "content":     doc.page_content[:200]    # preview only
        }
        for doc in source_docs
    ]

    return {"answer": answer, "sources": sources}


# Entry point
if __name__ == "__main__":
    # Connectivity Test
    print("\n--- Connecting to LLM via Groq ---")
    try:
        llm = get_llm()
        response = llm.invoke("Hello! Are you connected via Groq?")
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Error: {e}")