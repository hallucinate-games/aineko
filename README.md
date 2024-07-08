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

# Run as a server
(.venv) $ python src/main.py --server

```

## Endpoints

### `POST /add-dir`
This endpoint allows you to specify a directory to be recursively ingested into `aineko`'s embedding database
- Expects: JSON in body of the format:
  - 'dir_to_add': a string of the directory you'd like to have recursively ingested for indexing by `aineko`
- Returns: JSON in the format:
  - 'files_added': List of strings of paths of files ingested by `aineko`

### `POST /query`
This endpoint allows you to send a query to be matched by nearest embeddings.
- Expects: JSON in body of the format:
  - `query`: string query
- Returns: JSON (format TBD, right now we're just returning raw chromadb results as a hack)

### `GET file/*`
Downloads the file specified at `*`. Must be a URL encoded absolute path.

### `GET /`
nyaa!
