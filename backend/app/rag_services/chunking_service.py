"""
Chunking Service - Multiple chunking strategies for RAG
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import re
import tiktoken


@dataclass
class Chunk:
    """Represents a chunk of text with metadata"""
    text: str
    start_char: int
    end_char: int
    metadata: dict


class ChunkingStrategyBase(ABC):
    """Base class for chunking strategies"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """Split text into chunks"""
        pass
    
    def _count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count tokens using tiktoken"""
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            # Fallback to approximate character count
            return len(text) // 4


class FixedSizeChunkingStrategy(ChunkingStrategyBase):
    """Simple fixed-size chunking with overlap"""
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        chunks = []
        metadata = metadata or {}
        
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunking_strategy": "fixed_size"
                    }
                ))
            
            start = end - self.overlap if end < text_len else text_len
        
        return chunks


class RecursiveCharacterChunkingStrategy(ChunkingStrategyBase):
    """Recursive chunking that respects paragraph and sentence boundaries"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, 
                 separators: Optional[List[str]] = None):
        super().__init__(chunk_size, overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        metadata = metadata or {}
        return self._recursive_split(text, 0, metadata)
    
    def _recursive_split(self, text: str, start_offset: int, 
                         metadata: dict) -> List[Chunk]:
        """Recursively split text using separator hierarchy"""
        chunks = []
        
        if len(text) <= self.chunk_size:
            return [Chunk(
                text=text.strip(),
                start_char=start_offset,
                end_char=start_offset + len(text),
                metadata={
                    **metadata,
                    "chunk_index": 0,
                    "chunking_strategy": "recursive_character"
                }
            )]
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                current_chunk = ""
                current_start = start_offset
                
                for i, split in enumerate(splits):
                    # Add separator back except for last split
                    split_with_sep = split + (separator if i < len(splits) - 1 else "")
                    
                    if len(current_chunk) + len(split_with_sep) <= self.chunk_size:
                        current_chunk += split_with_sep
                    else:
                        if current_chunk:
                            chunks.append(Chunk(
                                text=current_chunk.strip(),
                                start_char=current_start,
                                end_char=current_start + len(current_chunk),
                                metadata={
                                    **metadata,
                                    "chunk_index": len(chunks),
                                    "chunking_strategy": "recursive_character"
                                }
                            ))
                            # Start new chunk with overlap
                            overlap_text = current_chunk[-self.overlap:] if self.overlap > 0 else ""
                            current_start = current_start + len(current_chunk) - len(overlap_text)
                            current_chunk = overlap_text + split_with_sep
                        else:
                            current_chunk = split_with_sep
                
                # Add remaining chunk
                if current_chunk.strip():
                    chunks.append(Chunk(
                        text=current_chunk.strip(),
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "chunking_strategy": "recursive_character"
                        }
                    ))
                
                return chunks
        
        # Fallback: no separator worked, use fixed-size
        return FixedSizeChunkingStrategy(self.chunk_size, self.overlap).chunk(text, metadata)


class SemanticChunkingStrategy(ChunkingStrategyBase):
    """Semantic chunking based on sentence boundaries"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        super().__init__(chunk_size, overlap)
        # Sentence boundary regex
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        chunks = []
        metadata = metadata or {}
        
        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        
        current_chunk = ""
        current_start = 0
        char_offset = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                char_offset += len(sentence) + 1
                continue
            
            # Check if adding sentence exceeds chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(Chunk(
                    text=current_chunk.strip(),
                    start_char=current_start,
                    end_char=current_start + len(current_chunk),
                    metadata={
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunking_strategy": "semantic",
                        "sentence_count": len(self.sentence_pattern.split(current_chunk))
                    }
                ))
                
                # Start new chunk with overlap (last sentence)
                if self.overlap > 0:
                    overlap_sentences = self.sentence_pattern.split(current_chunk)[-1:]
                    overlap_text = " ".join(overlap_sentences)
                    current_start = current_start + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_start = char_offset
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                    current_start = char_offset
            
            char_offset += len(sentence) + 1
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(Chunk(
                text=current_chunk.strip(),
                start_char=current_start,
                end_char=current_start + len(current_chunk),
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "chunking_strategy": "semantic",
                    "sentence_count": len(self.sentence_pattern.split(current_chunk))
                }
            ))
        
        return chunks


class DocumentAwareChunkingStrategy(ChunkingStrategyBase):
    """Document-aware chunking that respects headers and structure"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        super().__init__(chunk_size, overlap)
        # Patterns for markdown headers, bullet points, etc.
        self.header_pattern = re.compile(r'^#{1,6}\s+.+$', re.MULTILINE)
        self.bullet_pattern = re.compile(r'^[\*\-\+]\s+', re.MULTILINE)
    
    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        chunks = []
        metadata = metadata or {}
        
        # Split by headers first
        sections = self._split_by_headers(text)
        
        char_offset = 0
        for section_header, section_text in sections:
            # Add header context to metadata
            section_metadata = {
                **metadata,
                "section_header": section_header if section_header else "root"
            }
            
            # If section is small enough, keep as single chunk
            if len(section_text) <= self.chunk_size:
                chunks.append(Chunk(
                    text=section_text.strip(),
                    start_char=char_offset,
                    end_char=char_offset + len(section_text),
                    metadata={
                        **section_metadata,
                        "chunk_index": len(chunks),
                        "chunking_strategy": "document_aware"
                    }
                ))
                char_offset += len(section_text)
            else:
                # Split large sections using recursive strategy
                section_chunks = RecursiveCharacterChunkingStrategy(
                    self.chunk_size, self.overlap
                ).chunk(section_text, section_metadata)
                
                for chunk in section_chunks:
                    chunks.append(Chunk(
                        text=chunk.text,
                        start_char=char_offset + chunk.start_char,
                        end_char=char_offset + chunk.end_char,
                        metadata={
                            **chunk.metadata,
                            "chunk_index": len(chunks),
                            "chunking_strategy": "document_aware"
                        }
                    ))
                
                char_offset += len(section_text)
        
        return chunks
    
    def _split_by_headers(self, text: str) -> List[tuple]:
        """Split text into sections based on headers"""
        sections = []
        lines = text.split('\n')
        
        current_header = None
        current_section = []
        
        for line in lines:
            if self.header_pattern.match(line):
                # Save previous section
                if current_section:
                    sections.append((current_header, '\n'.join(current_section)))
                
                # Start new section
                current_header = line.strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append((current_header, '\n'.join(current_section)))
        
        return sections if sections else [(None, text)]


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies"""
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        chunk_size: int = 500,
        overlap: int = 50,
        **kwargs
    ) -> ChunkingStrategyBase:
        """Create a chunking strategy by type"""
        
        strategies = {
            "fixed_size": FixedSizeChunkingStrategy,
            "recursive_character": RecursiveCharacterChunkingStrategy,
            "semantic": SemanticChunkingStrategy,
            "document_aware": DocumentAwareChunkingStrategy
        }
        
        strategy_class = strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(
                f"Unknown strategy: {strategy_type}. "
                f"Available: {list(strategies.keys())}"
            )
        
        return strategy_class(chunk_size=chunk_size, overlap=overlap, **kwargs)
