# AINEKO
AINEKO is a dataset embedding storage and retrieval implementation for RAG (retrieval augmented generation) systems. In addition to indexing the data itself, it strategically indexes metadata, summaries, and other bits of information that augment the retrieval process and fix a lot of the downfalls that come with storing embeddings of chunks of data naively.

## Running
```
# Create a virtual environment
$ python -m venv .venv

# Activate virtual environment
# Unix machines:
$ source .venv/Scripts/activate
# Windows machines:
$ .venv/Scripts/activate

# Install requirments
(.venv) $ pip install -r requirements.txt

# Run
(.venv) $ python src/main.py

```
