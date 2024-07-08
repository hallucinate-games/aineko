import argparse
from pprint import pprint
import sys

import chromadb

from aineko import add_dir_to_collection, add_file_to_collection, AinekoEmbeddingFunction 

# Use this for persistent client
#chroma_client = chromadb.PersistentClient(path="../demo-db")
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())
# Uncomment these for persistent client
#chroma_client.delete_collection("aineko-demo")
#collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())

def main():
    parser = argparse.ArgumentParser(description="Smart embedding vector storage and retrieval")

    parser.add_argument('--file', help="A file to store the embeddings of.")
    parser.add_argument('--dir', help="Recursively store embeddings of files in this directory.")
    parser.add_argument('--query', help="Run this embedding search and output results")
    parser.add_argument('--server', help="Run aineko in server mode", action='store_true')

    args = parser.parse_args()

    file_to_add = getattr(args, 'file', None)
    dir_to_add = getattr(args, 'dir', None)
    query = getattr(args, 'query', None)
    server_mode = getattr(args, 'server', False)

    if not (file_to_add or dir_to_add or query or server_mode):
        parser.print_usage()
        sys.exit(1)
    
    if file_to_add:
        add_file_to_collection(collection=collection, file_path=file_to_add)
    if dir_to_add:
        add_dir_to_collection(collection=collection, dir=dir_to_add)
    if query:
        query_results = collection.query(
                query_texts=[query],
                n_results=3
        )
        pprint(query_results)
    if server_mode:
        raise NotImplementedError("Server mode is not yet implemented")

if __name__ == "__main__":
    main()
