from document_loader import load_pdf_to_vectorstore
from agent import create_research_agent

if __name__ == "__main__":
    file_path = "research.pdf"  # Place your PDF here
    vectorstore = load_pdf_to_vectorstore(file_path)

    agent = create_research_agent(vectorstore)

    print("🔍 Ask your research question:")
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break
        response = agent.run(query)
        print("🤖 Bot:", response)
