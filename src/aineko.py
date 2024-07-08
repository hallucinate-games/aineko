from dataclasses import dataclass
import os
import time
from typing import Generator

import chromadb
from chromadb.api.types import D, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from nltk import download
from nltk.tokenize import sent_tokenize

_punkt_downloaded = False
_chroma_client = None
_collection = None

chroma_client = chromadb.Client()


class AinekoEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: D) -> Embeddings:
        # Currently a pass-through function. Leaving it here in case we need to
        # inject functionality at the embedding layer.
        embeddings = embedding_functions.DefaultEmbeddingFunction()(input)
        return embeddings


def create_collection(name: str=None, persistent: bool=False):
    global _chroma_client
    global _collection
    name = name or 'aineko-demo'
    if persistent:
        _chroma_client = chromadb.PersistentClient(path="../demo-db")
    else:
        _chroma_client = chromadb.Client()
    _collection = chroma_client.get_or_create_collection(name=name, embedding_function=AinekoEmbeddingFunction())

    if persistent:
        _chroma_client.delete_collection(name)
        _collection = chroma_client.get_or_create_collection(name="aineko-demo", embedding_function=AinekoEmbeddingFunction())
    return _collection


def get_collection():
    global _collection
    if not _collection:
        return create_collection()
    return _collection
    

def add_file_to_collection(file_path: str):
    """ Add chunks of a file to a chromadb collection """
    document_chunks = _generate_overlapping_chunks(file_path)
    chunks_added = 0
    for chunk in document_chunks:
        _collection.add(
            ids=[f"chunk{chunk.chunk_idx}:{file_path}"],
            documents=[chunk.text],
            metadatas=[chunk.generate_metadata_object()]
        )
        chunks_added += 1
    maybe_plural_chunks = 'chunk' if chunks_added == 1 else 'chunks'
    print(f"Added {chunks_added} {maybe_plural_chunks} to collection from file {file_path}")


def add_dir_to_collection(dir: str):
    files_added = []
    for root, _, files in os.walk(dir):
        for file in files:
            file_path = os.path.join(root, file)
            add_file_to_collection(file_path)
            files_added.append(file_path)
    return files_added


def _sentence_chunk_file(file_path: str) -> Generator[str, None, None]:
    """ Breaks a file into approximately sentence sized chunks. """
    global _punkt_downloaded
    if not _punkt_downloaded:
        # Used by `_sentence_chunk_file` for sentence tokenization
        try:
            download('punkt')
            _punkt_downloaded = True
        except Exception as exc:
            print(
                "While you do not need an internet connection to run aineko, "
                "you do need to be connected to the internet the first time you run "
                "it to download the sentence tokenization system.")
            raise exc
    with open(file_path, 'r') as f:
        raw_text = f.read()
        
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(raw_text)
    
    current_chunk = ''
    for sentence in sentences:
        if len(sentence) < 5:
            current_chunk += ' ' + sentence
        else:
            yield current_chunk
            current_chunk = sentence
    yield current_chunk


def _get_file_times(file_path):
    # Get the creation time
    creation_time = os.path.getctime(file_path)
    # Get the last modification time
    modification_time = os.path.getmtime(file_path)

    # Convert the timestamps to readable format
    creation_time = time.ctime(creation_time)
    modification_time = time.ctime(modification_time)

    return creation_time, modification_time


@dataclass
class DocumentChunk:
    text: str
    file_path: str
    begin_sentence_idx: int
    end_sentence_idx: int
    chunk_idx: int
    file_created_at: str
    file_last_updated_at: str

    def text_with_metadata(self):
        return f"[File] {self.file_path}\n[Created at] {self.file_created_at}\n[Last updated at] {self.file_last_updated_at}\n[Chunk contents]\n{self.text}"

    def generate_metadata_object(self):
        return {
            "file_path": self.file_path,
            "chunk_index": self.chunk_idx,
            "file_created_at": self.file_created_at,
            "file_last_updated_at": self.file_last_updated_at
        }


def _generate_overlapping_chunks(
        file_path: str,
        chunk_size: int = 4,
        overlap: int = 1,
        ) -> Generator[DocumentChunk, None, None]:
    """ Generates overlapping chunks from file packaged with metadata. """
    sentence_chunks = _sentence_chunk_file(file_path)
    file_created_at, file_last_updated_at = _get_file_times(file_path)
    current_chunk_size = 0
    current_chunk_idx = 0
    current_chunk = ''
    overlap_chunk_buffer = []
    for sentence_number, sentence in enumerate(sentence_chunks):
        overlap_chunk_buffer.append(sentence)
        if len(overlap_chunk_buffer) > overlap:
            overlap_chunk_buffer.pop(0)
        current_chunk += ' ' + sentence
        current_chunk_size += 1
        if current_chunk_size >= chunk_size:
            yield DocumentChunk(
                text=current_chunk,
                file_path=file_path,
                begin_sentence_idx=sentence_number - current_chunk_size,
                end_sentence_idx=sentence_number,
                chunk_idx=current_chunk_idx,
                file_created_at=file_created_at,
                file_last_updated_at=file_last_updated_at,
            )
            current_chunk_idx += 1
            current_chunk = ' '.join(overlap_chunk_buffer)
            current_chunk_size = len(overlap_chunk_buffer)
    if current_chunk.strip() != '' and (current_chunk_size != len(overlap_chunk_buffer) or current_chunk_idx == 0):
        yield DocumentChunk(
            text=current_chunk,
            file_path=file_path,
            begin_sentence_idx=sentence_number - current_chunk_size,
            end_sentence_idx=sentence_number,
            chunk_idx=current_chunk_idx,
            file_created_at=file_created_at,
            file_last_updated_at=file_last_updated_at,
        )
