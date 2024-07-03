from pprint import pprint
import sys

import chromadb

from aineko import add_file_to_collection, AinekoEmbeddingFunction 

# Use this for persistent client
#chroma_client = chromadb.PersistentClient(path="../demo-db")
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())
# Uncomment these for persistent client
#chroma_client.delete_collection("aineko-demo")
#collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <file_path> <query>")
        sys.exit(1)

    file_path = sys.argv[1]
    query = sys.argv[2]

    add_file_to_collection(collection=collection, file_path=file_path)
    
    query_results = collection.query(
        query_texts=[query],
        n_results=3
    )
    pprint(query_results)


if __name__ == "__main__":
    main()
