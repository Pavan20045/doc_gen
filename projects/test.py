def extract_text_from_pdf(file_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
text = extract_text_from_pdf("D:/Bajaj Hackathon/Web app/uploads/1754572090107-Document 3.pdf")
print("Extracted text length:", len(text))
