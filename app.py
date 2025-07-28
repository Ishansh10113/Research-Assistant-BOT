import streamlit as st
import os
import tempfile
from document_loader import load_to_vectorstore
from agent import create_conversational_agent

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS for a cleaner UI ---
st.markdown("""
<style>
    /* Main chat container */
    .st-emotion-cache-1jicfl2 {
        padding-top: 2rem;
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    /* Chat input box */
    .st-emotion-cache-135i1ga {
        background-color: #f0f2f6;
    }
    /* Hide the Streamlit header/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- Session State Initialization ---
def initialize_session_state():
    """Initializes session state variables if they don't exist."""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False

initialize_session_state()


# --- Sidebar UI ---
with st.sidebar:
    st.header("🧠 AI Research Assistant")

    if st.button("New Chat", use_container_width=True):
        # Reset the chat but keep the agent
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("Upload Your Document")
    uploaded_file = st.file_uploader(
        "Upload a document to start a conversation.",
        type=["pdf", "docx", "pptx", "xlsx", "csv", "zip"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if not st.session_state.document_processed:
            with st.status("Processing Document...", expanded=True) as status:
                try:
                    # Save to a temporary file
                    suffix = "." + uploaded_file.name.split('.')[-1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    st.write("📄 Loading and splitting document...")
                    vectorstore = load_to_vectorstore(tmp_path)

                    st.write("🤖 Creating conversational agent...")
                    st.session_state.agent = create_conversational_agent(vectorstore)
                    st.session_state.document_processed = True
                    st.session_state.messages = [] # Reset messages for new doc

                    # Clean up the temporary file
                    os.remove(tmp_path)
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)

                except Exception as e:
                    status.update(label=f"❌ Error: {e}", state="error")
                    st.session_state.document_processed = False

    st.markdown("---")
    answer_length = st.radio(
        "**Select Answer Length**",
        ("Concise", "Detailed"),
        index=1,
        horizontal=True,
    )
    st.markdown("---")
    st.info("Built using Streamlit, LangChain, Groq by Ishansh.")


# --- Main Chat Interface ---
st.header("Chat with your Document")

# Display intro message if no document is processed
if not st.session_state.document_processed:
    st.info("Please upload a document in the sidebar to begin.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in the message
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(message["sources"], 1):
                     st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'N/A')}` | **Page:** `{doc.metadata.get('page', 'N/A')}`")
                     st.markdown(doc.page_content)


# Accept user input
if prompt := st.chat_input("Ask a question..." if st.session_state.document_processed else "Upload a document to start..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            try:
                # --- THE FIX IS HERE ---
                # Format the chat history into a list of tuples (human, ai)
                formatted_history = []
                for msg in st.session_state.messages[:-1]: # Exclude the current user prompt
                    if msg["role"] == "user":
                        # Find the corresponding assistant message
                        assistant_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "assistant" and st.session_state.messages.index(m) > st.session_state.messages.index(msg)), None)
                        if assistant_msg:
                            formatted_history.append((msg["content"], assistant_msg))

                # Prepare the question
                full_question = f"Provide a {answer_length.lower()} answer. Question: {prompt}"

                # Run the agent
                response = st.session_state.agent(
                    {"question": full_question, "chat_history": formatted_history}
                )
                answer = response.get("answer", "Sorry, I encountered an issue.")
                sources = response.get("source_documents", [])

                # Display the response and sources
                st.markdown(answer)
                if sources:
                    with st.expander("📚 View Sources"):
                         for i, doc in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:** `{doc.metadata.get('source', 'N/A')}` | **Page:** `{doc.metadata.get('page', 'N/A')}`")
                            st.markdown(doc.page_content)

                # Add assistant response and sources to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})