from dataclasses import dataclass
import os
import time
from typing import Generator

from chromadb.api.types import D, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from nltk import download
from nltk.tokenize import sent_tokenize

# Used by `_sentence_chunk_file` for sentence tokenization
try:
    download('punkt')
except Exception as exc:
    print(
        "While you do not need an internet connection to run aineko, "
        "you do need to be connected to the internet the first time you run "
        "it to download the sentence tokenization system.")
    raise exc

class AinekoEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: D) -> Embeddings:
        # Currently a pass-through function. Leaving it here in case we need to
        # inject functionality at the embedding layer.
        embeddings = embedding_functions.DefaultEmbeddingFunction()(input)
        return embeddings

def add_file_to_collection(collection, file_path: str):
    """ Add chunks of a file to a chromadb collection """
    document_chunks = _generate_overlapping_chunks(file_path)
    chunks_added = 0
    for chunk in document_chunks:
        collection.add(
            ids=[f"chunk{chunk.chunk_idx}:{file_path}"],
            documents=[chunk.text],
            metadatas=[chunk.generate_metadata_object()]
        )
        chunks_added += 1
    maybe_plural_chunks = 'chunk' if chunks_added == 1 else 'chunks'
    print(f"Added {chunks_added} {maybe_plural_chunks} to collection from file {file_path}")

def _sentence_chunk_file(file_path: str) -> Generator[str, None, None]:
    """ Breaks a file into approximately sentence sized chunks. """
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
    if current_chunk_size != len(overlap_chunk_buffer):
        yield DocumentChunk(
            text=current_chunk,
            file_path=file_path,
            begin_sentence_idx=sentence_number - current_chunk_size,
            end_sentence_idx=sentence_number,
            chunk_idx=current_chunk_idx,
            file_created_at=file_created_at,
            file_last_updated_at=file_last_updated_at,
        )
