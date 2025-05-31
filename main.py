import os
from dotenv import load_dotenv
from google import genai
from google.genai.types import File
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent import create_financial_agent

# Necessary Parameters
LLM_MODEL_NAME = "gemini-2.0-flash"
DATA_DIR = "data"
uploaded_file_details = None   # To store URI and mime_type

# Set up Genai SDK and LLM
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# GenAI SDK Client Setup - set as a global var
client = genai.Client(api_key=gemini_api_key)



# Initialize Gemini for LangChain and GenAI SDK
def initialize_llm():
    """
    Initializes and returns the Gemini LLM with streaming enabled.
    """
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found. Make sure it's set in your .env file.")
        return None
    
    # GenAI SDK Initialization
    try:
        if client:
            print("Google GenAI SDK configured successfully for file operations.")
    except Exception as e:
        print(f"Error configuring Google GenAI SDK: {e}")
        print("File Uploading might not work. Ensure GEMINI_API_KEY is valid.")
        return None
    
    # LangChain Initialization
    try:
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            google_api_key=gemini_api_key,
            temperature=0.2,
            disable_streaming=False
        )
        print(f"LangChain Gemini LLM ({LLM_MODEL_NAME}) initialized successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing Gemini LLM ({LLM_MODEL_NAME}): {e}")
        print("Please check your API key, model name, and Internet connection.")
        return None
    

# Upload PDF file to Gemini
def upload_pdf_to_gemini(pdf_path: str, display_name: str) -> File | None:
    """
    Uploads the PDF file to Gemini and returns File object.
    """
    print(f"\nUploading '{display_name}' to Gemini... This may take a moment.")
    try:
        pdf_file = client.files.upload(file=pdf_path)
        print(f"File uploaded successfully. URI: {pdf_file.uri}")
        return pdf_file
    except Exception as e:
        print(f"Error uploading PDF to Gemini: {e}")
        print("Please check API key permissions for file uploading and network connection.")
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
    # global uploaded_file_details

    # LLM and SDK Initialization
    llm = initialize_llm()
    if not llm:
        print("LLM or GenAI SDK initialization failure. Exiting...")
        return
    
    # PDF Loading
    pdf_path = get_pdf_path_from_user()
    if not pdf_path:
        print("No PDF file selected. Exiting...")
        return

    # Upload PDF to Gemini
    pdf_display_name = os.path.basename(pdf_path)
    gemini_file_object = upload_pdf_to_gemini(pdf_path, pdf_display_name)
    if not gemini_file_object:
        print("PDF Upload Failure. Exiting...")
        return
    
    # print("DEBUG: Uploaded file object:", vars(gemini_file_object))

    # Wrap file object and display name together
    file_context = {
        "file": gemini_file_object,
        "display_name": pdf_display_name
    }
    
    # Store URI and MIME type for tools
    # uploaded_file_details = {
    #     "uri": gemini_file_object.uri,
    #     "mime_type": gemini_file_object.mime_type,
    #     "display_name": gemini_file_object.display_name
    # }

    # Agent Creation. Passing file details
    financial_agent_executor = create_financial_agent(llm, file_context)
    if not financial_agent_executor:
        print("Failed to create the financial agent. Exiting...")
        return
    
    print(f"\nFinancial Report Analyzer Agent is ready to discuss '{pdf_display_name}'.")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Ask questions about the loaded financial report.")
    print("-" * 50)

    # For now, the tools will get the URI and reference it in their prompts
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ["exit", "quit"]:
            # Clean up the uploaded file from Gemini
            if gemini_file_object and gemini_file_object.uri:
                try:
                    print(f"\nAttempting to delete uploaded file: {pdf_display_name} ({gemini_file_object.uri})")
                    client.files.delete(name=gemini_file_object.name)
                    print("File deleted successfully from Gemini.")
                except Exception as e:
                    print(f"Could not delete file from Gemini: {e}. It may be auto-deleted later by Google.")
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