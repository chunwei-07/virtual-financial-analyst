import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from google.api_core.exceptions import GoogleAPIError
from langchain_core.exceptions import OutputParserException

from src.pdf_processor import extract_text_from_pdf
from src.agent import create_financial_agent

# Configure LLM model name
LLM_MODEL_NAME = "gemini-2.0-flash-exp"
DATA_DIR = "data"

# Initialize Gemini
def initialize_llm():
    """
    Initializes and returns the Gemini LLM with streaming enabled.
    """
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found. Make sure it's set in your .env file.")
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=gemini_api_key,
            temperature=0.2,
            disable_streaming=False
        )
        print(f"Gemini LLM ({LLM_MODEL_NAME}) initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Gemini LLM ({LLM_MODEL_NAME}): {e}")
        print("Please check your API key, model name, and Internet connection.")
        return None
    
def get_pdf_path_from_user() -> str | None:
    """
    Prompts the user for a PDF filename and validates its existence in DATA_DIR.
    Loops until a valid file is provided or the user quits.
    """
    # Ensure data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found. Please create it and add your PDF files there.")
        return None
    if not os.path.isdir(DATA_DIR):
        print(f"Error: '{DATA_DIR}' exists but is not a directory.")
        return None
    
    while True:
        pdf_file_name = input(
            f"\nEnter the name of the PDF file in the '{DATA_DIR}' folder "
            f"(e.g., Meta_Q1_2024_Earnings.pdf), or type 'quit' to exit: "
        )
        if pdf_file_name.lower() == 'quit':
            return None
        
        if not pdf_file_name.lower().endswith(".pdf"):
            pdf_file_name += ".pdf"   # Auto append .pdf if missing
            print(f"Assuming you meant: {pdf_file_name}")

        pdf_path = os.path.join(DATA_DIR, pdf_file_name)

        if os.path.exists(pdf_path) and os.path.isfile(pdf_path):
            return pdf_path
        else:
            print(f"Error: PDF file not found at '{pdf_path}'. Please check the filename and ensure it's in the '{DATA_DIR}' folder.")

            # List available PDF files in data directory
            try:
                pdf_files_in_data = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
                if pdf_files_in_data:
                    print("\nAvailable PDF files in data directory:")
                    for f_name in pdf_files_in_data:
                        print(f"  - {f_name}")
                    print("")   # newline for spacing
                else:
                    print(f"No PDF files found in '{DATA_DIR}'.")
            except OSError as e:
                print(f"Could not list files in '{DATA_DIR}': {e}")

# Main function
def main_conversational_loop():
    llm = initialize_llm()
    if not llm:
        print("Exiting due to LLM initialization failure.")
        return
    
    # PDF Loading
    pdf_path = get_pdf_path_from_user()
    if not pdf_path:
        print("No PDF file selected. Exiting...")
        return

    print(f"\nProcessing PDF: {pdf_path}")
    extracted_text = extract_text_from_pdf(pdf_path)
    if not extracted_text:
        print("Failed to extract text from PDF. Exiting...")
        return
    
    # CONTEXT WINDOW CONSIDERATION
    # The entire extracted_text may be too large for LLM context window
    # For now, the agent's placeholder "DocumentSearch" tool will get a small excerpt.
    # The LLM will implicitly have access to what fits in its context when the agent runs.
    # A more robust solution would involved summarization or RAG

    print(f"Successfully extracted {len(extracted_text)} characters from the PDF.")
    print("The agent will use this document for answering your questions.")

    # Agent Creation
    financial_agent_executor = create_financial_agent(llm, extracted_text)
    if not financial_agent_executor:
        print("Failed to create the financial agent. Exiting...")
        return
    
    print("\nFinancial Report Analyzer Agent is ready.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Ask questions about the loaded financial report.")
    print("-" * 50)
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting agent. Goodbye!")
            break
        if not user_query.strip():
            continue

        print("\nAgent: ", end="", flush=True)   # Print "Agent: " once, no newline, flush buffer
        
        try:
            # Use agent_executor.stream() for streaming responses
            full_response_content = ""
            for chunk in financial_agent_executor.stream({"input": user_query}):
                if "output" in chunk:  # Typical for final answer from AgentExecutor
                    print(chunk["output"], end="", flush=True)
                    full_response_content += chunk["output"]
            print()   # Add a newline after full response is streamed

        except OutputParserException as ope:
            print("\n\n[Agent Error] I had trouble understanding the last response structure. Let's try that again, or rephrase your query.")
            print(f"Details: {ope}")
        except GoogleAPIError as gae:
            print(f"\n\n[API Error] A Google API error occurred: {gae.message}")
            print("This could be due to network issues, API key problems, or rate limits.")
            if "PERMISSION_DENIED" in str(gae) or "API key not valid" in str(gae):
                print("Please double check your GEMINI_API_KEY and ensure it has the correct permissions.")
        except ConnectionError as ce:
            print(f"\n\n[Network Error]: Could not connect: {ce}")
            print("Please check your Internet connection.")
        except Exception as e:
            print(f"\n\n[Unexpected Error] An unexpected error occurred: {e}")
            print("If this persists, please report the issue.")

    print("\nSession Ended.")


# Check Gemini Connection
def check_llm_connection():
    """
    Checks if the connection to the Gemini LLM can be established.
    """
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Error: GEMINI_API_KEY not found in .env file or environment variables.")
            return
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                     google_api_key=gemini_api_key,
                                     temperature=0.3)
        
        print("Attempting to connect to Gemini LLM...")
        response = llm.invoke("Hello! What is your name?")
        print("Successfully connected to Gemini LLM.")
        print(f"LLM Response: {response.content}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your API key, Internet connection, and model availability.")

if __name__ == "__main__":
    main_conversational_loop()