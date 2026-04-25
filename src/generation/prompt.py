from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate

# Setting up the system message
SYSTEM_MESSAGE = """
    You are a precise and helpful document assistant.
    Your job is to answer the user's question using ONLY the context provided below.

    Rules you must follow:
    1. Answer strictly from the context — do not use outside knowledge.
    2. If the answer is not in the context, say exactly:
    "I could not find the answer in the provided document."
    3. Be concise and direct — no filler phrases like "Based on the context..."
    4. If the context contains numbers, dates, or names — use them exactly as written.
    5. Never make up information.
"""

def get_rag_chat_prompt() -> ChatPromptTemplate:
    system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_MESSAGE)
    human_template = """
    Context:
    ---------
    {context}
    ---------

    Question: {question}

    Answer:
    """
    return ChatPromptTemplate.from_messages([system_prompt, human_template])

def get_rag_instruct_prompt() -> PromptTemplate:
    template = """
            You are a precise document assistant. Answer using ONLY the context below.
            If the answer is not present, say "I could not find the answer in the provided document."

            Context:
            ---------
            {context}
            ---------

            Question: {question}

            Answer:
        """
    return PromptTemplate(input_variables=["context", "question"], template=template)

# Condense question prompt
CONDENSE_QUESTION_TEMPLATE = """
Given the following conversation history and a follow-up question,
rephrase the follow-up question to be a fully standalone question
that can be understood without the conversation history.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

def get_condense_question_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["chat_history", "question"], template=CONDENSE_QUESTION_TEMPLATE)

def get_selector(mode: str = "chat") -> PromptTemplate | ChatPromptTemplate:
    if mode == "chat":
        return get_rag_chat_prompt()
    elif mode == "instruct":
        return get_rag_instruct_prompt()
    else:
        raise ValueError(f"Unknown prompt mode: {mode}")

if __name__ == "__main__":
    prompt = get_rag_chat_prompt()
    sample = prompt.format_messages(
        context="Apple reported revenue of $94.8B in Q1 2024.",
        question="What was Apple's revenue in Q1 2024?"
    )
    for msg in sample:
        print(f"[{msg.type.upper()}]\n{msg.content}\n")
