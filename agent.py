from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def create_conversational_agent(vectorstore):
    """
    Creates a conversational retrieval agent.

    This agent can answer questions based on the provided vectorstore and remembers
    the conversation history to answer follow-up questions.
    """
    llm = ChatGroq(
        temperature=0,
        model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    # Create a conversational retrieval chain.
    # IMPORTANT: Set verbose=False to prevent the agent from returning its internal thoughts.
    conversation_agent = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        verbose=False  # <-- THE FIX IS HERE
    )

    return conversation_agent