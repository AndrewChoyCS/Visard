import pdfplumber

class ParagraphChunker:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def chunk_by_paragraph(self): 
        paragraphs = []
        
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    page_paragraphs = text.split('\n\n')  
                    paragraphs.extend(page_paragraphs)
        
        return paragraphs

chunker = ParagraphChunker("./data/onepage.pdf")
paragraphs = chunker.chunk_by_paragraph()

print("----- Paragraphs -----")
for i, paragraph in enumerate(paragraphs, start=1):
    print(f"Paragraph {i}:\n{paragraph}\n{'-' * 20}")