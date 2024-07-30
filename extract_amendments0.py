import os
import fitz  # PyMuPDF
import tiktoken
import re
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_file):
    if pdf_file is None:
        raise ValueError("No PDF file provided")
    document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    if document.page_count == 0:
        raise ValueError("The provided PDF file is empty")
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def split_text_into_token_chunks(text, max_tokens, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def generate_prompt():
    return """
    You are an AI language model tasked with extracting amendments related to SEBI Mutual Funds Regulations, 1996.
    Don't even try to add your creativity. Just return the exact thing as it is.
    Focus on the following pattern:
    1. After regulation X, the following regulation shall be inserted, namely, — "Regulation Text".
    2. In the Schedule, in clause Y, after the words "Existing Text" the words "Inserted Text" shall be added.

    Ignore non-English parts.

    Here is the amendment text:
    """

def search_for_amendments(text):
    pattern1 = r"after regulation \d+[A-Z]?, the following regulation shall be inserted, namely, — .*"
    pattern2 = r"in the .* Schedule, in clause .* after the words .* the words .* shall be added"

    matches = re.findall(f"({pattern1}|{pattern2})", text, re.DOTALL)
    return matches

def main(pdf_file):
    text = extract_text_from_pdf(pdf_file)
    
    amendment_texts = search_for_amendments(text)
    if not amendment_texts:
        return "No amendments related to SEBI Mutual Funds Regulations, 1996 were found."
    else:
        amendments = []
        
        for amendment in amendment_texts:
            prompt = generate_prompt() + amendment
            response = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": "Extract amendments related to SEBI Mutual Funds Regulations, 1996."},
                    {"role": "user", "content": prompt},
                ]
            )
            amendments.append(response.choices[0].message.content)
        
        formatted_amendment = "\n\n".join(amendments)
        return formatted_amendment
