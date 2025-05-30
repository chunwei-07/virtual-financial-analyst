import fitz
import os

def extract_text_from_pdf(pdf_path: str) -> str | None:
    """
    Extracts all text content from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str | None: The concatenated text from all pages of the PDF,
                    or None if the file doesn't exist or an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    
    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text.append(page.get_text("text"))  # "text" for plain text extraction
        doc.close()
        return "\n".join(full_text)
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return None
    
if __name__ == "__main__":
    # This is to test the function directly
    sample_pdf_path = os.path.join("..", "data", "Meta_Q1_2024_Earnings.pdf")

    if not os.path.exists(sample_pdf_path):
        print(f"Test PDF not found: {sample_pdf_path}")
        print("Please download a sample PDF")
        print("and place it in the 'data' folder, then update the path if necessary.")
    else:
        print(f"Extracting text from: {sample_pdf_path}")
        extracted_text = extract_text_from_pdf(sample_pdf_path)

        if extracted_text:
            print("\n--- Extracted Text (First 500 characters) ---")
            print(extracted_text[:500])
            print("\n--- End of Sample ---")
            print(f"\nTotal characters extracted: {len(extracted_text)}")
        else:
            print("No text extracted or an error occurred.")

        