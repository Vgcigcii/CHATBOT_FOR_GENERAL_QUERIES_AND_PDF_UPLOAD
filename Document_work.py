import easygui
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os

def accepting_user_file():
    print("accepting_user_file() called")
    print("Please upload only pdf format")
    try:
        file_path = easygui.fileopenbox(title = "Select your input file")
        if not file_path:
            print("Upload cancelled")
            return None
        if not file_path.endswith(".pdf"):
            print("Uploaded file is not a pdf file")
            return None
        print(f"File selected: {os.path.basename(file_path)}")
        # read and extract text from PDF
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            print("the file is encrypted")
            return None
        # extract text with page numbers for context
        full_text = ""
        for page_num,page in enumerate(reader.pages,1):
            page_text = page.extract_text()
            if page_text:
                full_text += f"[Page {page_num}] {page_text}\n\n"
        if not full_text.strip():
            print("No text could be extracted. The PDF may be scanned or image-based.")
            print("Consider using OCR tools like Tesseract for scanned PDFs.")
            return None
        print(f"Extracted text:{len(full_text)} characters from {len(reader.pages)} pages")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )
        chunks = splitter.split_text(full_text)
        print(f"Your file has been split into  chunks")
        if chunks:
            preview = chunks[0][:200]+"..." if len(chunks[0]) > 200 else chunks[0]
            print(f"first chunk Preview: {preview}")
        return chunks
    except FileNotFoundError:
        print("Error: File not found. Please select a valid file.")
        return None
    except PermissionError:
        print("Error: Permission denied. Make sure the file is not open elsewhere.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
