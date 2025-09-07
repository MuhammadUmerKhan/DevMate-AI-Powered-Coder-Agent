from dotenv import load_dotenv
from langchain.globals import set_verbose, set_debug
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI

from graph import create_workflow
from tools import init_project_root

def main():
    # Load environment variables
    load_dotenv()

    # Set LangChain debugging
    set_debug(True)
    set_verbose(True)

    # Initialize project root
    init_project_root()

    # Configuration: Modify these to change LLM or prompt
    llm_choice = "openai"  # Options: "openai" or "groq"
    user_prompt = "Build a colourful modern calculator in html css and js"

    # Select LLM based on choice
    if llm_choice == "groq":
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Create and run the workflow
    agent = create_workflow(llm)
    result = agent.invoke(
        {"user_prompt": user_prompt},
        {"recursion_limit": 100}
    )
    print("Final State:", result)

if __name__ == "__main__":
    main()