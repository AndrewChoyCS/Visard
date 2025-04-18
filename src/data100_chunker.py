import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd 

def smart_split(document):
    # Split by headers, callouts, code blocks, or LaTeX markers
    document = str(document)
    chunks = re.split(r'(?=^##|\n:::)', document)
    print(len(chunks))
    # Clean up
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def intelligent_chunker(segments, threshold=0.75):
    # Don't call smart_split here
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(segments)

    if len(embeddings) < 2:
        raise ValueError("Not enough segments to calculate similarity. Add more content.")

    similarities = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()

    breakpoints = [0]
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)
    breakpoints.append(len(segments))

    final_chunks = []
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i+1]
        joined = '\n\n'.join(segments[start:end])
        final_chunks.append(joined)

    return final_chunks


filePath = "data/data100/gradient_descent.qmd"
try: 
    with open(filePath, 'r', encoding='utf-8') as file:
        content = file.read()
except FileNotFoundError:
    print(f"Error: File not found at {filePath}")
except Exception as e:
    print(f"An error occurred: {e}") 

# print(content)
splitter = smart_split(content)
chinked = intelligent_chunker(splitter)
# print(chinked)
output_path = "data/data100/chunked_gradient_descent.qmd"

try:
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, chunk in enumerate(chinked, start=1):
            f.write(f"\n\n--- CHUNK {idx} ---\n\n")
            f.write(chunk.strip())
            f.write("\n")
    print(f"Chunks saved to: {output_path}")
except Exception as e:
    print(f"An error occurred while writing the chunks: {e}")
