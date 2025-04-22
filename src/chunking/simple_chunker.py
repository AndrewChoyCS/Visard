import re
import json
import re
import json

def remove_python_code_blocks(text):
    return re.sub(r'```{python}.*?```', '', text, flags=re.DOTALL)

def split_qmd_by_subsection(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    content = remove_python_code_blocks(content)
    chunks = re.split(r'(?=^###)', content, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, chunk in enumerate(chunks, start=1):
                f.write(f"\n\n--- CHUNK {idx} ---\n\n")
                f.write(chunk)
                f.write("\n")
        print(f"Chunks saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred while writing the chunks: {e}")

def split_qmd_to_json(file_path, output_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    content = remove_python_code_blocks(content)
    chunks = re.split(r'(?=^###)', content, flags=re.MULTILINE)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    chunk_dict = {str(idx): chunk for idx, chunk in enumerate(chunks, start=1)}

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_dict, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred while writing the JSON: {e}")


# simple_split_qmd_by_subsection(
#     "/Users/andrewchoy/Desktop/CS Projects/Visard/data/data100/feature_engineering.qmd",
#     "/Users/andrewchoy/Desktop/CS Projects/Visard/data/data100/chunked_feature_engineering.qmd"
# )

split_qmd_to_json(
    "/Users/andrewchoy/Desktop/CS Projects/Visard/data/data100/feature_engineering.qmd",
    "/Users/andrewchoy/Desktop/CS Projects/Visard/data/data100/chunked_feature_engineering.json"
)