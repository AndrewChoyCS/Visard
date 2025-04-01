import pdfplumber
import docx
import markdownify
import subprocess
import os

class FileConverter:
    def __init__(self, file_path):
        self.file_path = file_path

    def convert_to_markdown(self):
        # Handle different file types
        file_extension = os.path.splitext(self.file_path)[1].lower()

        if file_extension == '.pdf':
            return self._convert_pdf_to_markdown()
        elif file_extension == '.docx':
            return self._convert_docx_to_markdown()
        elif file_extension == '.html':
            return self._convert_html_to_markdown()
        elif file_extension == '.txt':
            return self._convert_txt_to_markdown()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _convert_pdf_to_markdown(self):
        text = ""
        with pdfplumber.open(self.file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        # Simple text conversion (could be improved with proper formatting)
        return text.replace("\n", "  \n")  # Markdown uses two spaces for line breaks

    def _convert_docx_to_markdown(self):
        doc = docx.Document(self.file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def _convert_html_to_markdown(self):
        with open(self.file_path, 'r') as file:
            html_content = file.read()
        return markdownify.markdownify(html_content)

    def _convert_txt_to_markdown(self):
        with open(self.file_path, 'r') as file:
            text = file.read()
        return text

class MarkdownChunker:
    def __init__(self, markdown_content, chunk_size=1000):
        self.markdown_content = markdown_content
        self.chunk_size = chunk_size

    def chunk_by_size(self):
        # Split the content by size into manageable chunks
        chunks = []
        start = 0
        while start < len(self.markdown_content):
            end = start + self.chunk_size
            chunks.append(self.markdown_content[start:end].strip())
            start = end
        return chunks

    def chunk_by_paragraph(self):
        # Split content by paragraphs (double newlines)
        paragraphs = self.markdown_content.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds the chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
            else:
                current_chunk += paragraph + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks
    

class FileProcessingPipeline:
    def __init__(self, file_path):
        self.file_path = file_path

    def process(self):
        # Step 1: Convert the file to markdown format
        converter = FileConverter(self.file_path)
        markdown_content = converter.convert_to_markdown()

        # Step 2: Chunk the markdown content into manageable parts
        chunker = MarkdownChunker(markdown_content)
        chunks = chunker.chunk_by_paragraph()  # You can use chunk_by_size() if preferred

        return chunks

# Example usage
file_path = './data/onepage.pdf'  # Change this to your file path
pipeline = FileProcessingPipeline(file_path)
chunks = pipeline.process()

# Print out the chunks
for idx, chunk in enumerate(chunks, start=1):
    print(f"Chunk {idx}:\n{chunk}\n{'-' * 20}")
