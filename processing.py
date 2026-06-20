import re
import unicodedata
import tiktoken
import ftfy
from cleantext import clean
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from langchain_text_splitters import RecursiveCharacterTextSplitter


def structural_clean(text: str) -> str:
    
    text = ftfy.fix_text(text)

    
    text = re.sub(r'<[^>]+>', ' ', text)

   
    import html
    text = html.unescape(text)

    
    text = unicodedata.normalize("NFKC", text)

    
    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)

   
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()



def remove_noise(text: str) -> str:
    lines = text.split('\n')
    seen = set()
    deduped = []

    for line in lines:
        normalized = line.strip().lower()
        
        
        if not normalized:
            deduped.append('')
            continue
        if len(normalized) < 10:          # Too short to be meaningful
            continue
        if normalized in seen:            # Exact duplicate line
            continue

        seen.add(normalized)
        deduped.append(line)

    return '\n'.join(deduped)


def redact_pii(text: str) -> str:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    results = analyzer.analyze(text=text, language='en')
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)

    return anonymized.text   



FILLER_PATTERNS = [
    r'it is (important|worth noting|crucial) to note that\s*',
    r'as (previously|mentioned|noted) (mentioned|above|before),?\s*',
    r'in (conclusion|summary),?\s*(it is clear that)?\s*',
    r'for the (purposes|sake) of (this|clarity),?\s*',
]

def compress_text(text: str) -> str:
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    
    text = re.sub(r'\|[-: ]+\|[-: |]+\n', '', text)

    
    text = re.sub(r'\.{2,}', '…', text)
    text = re.sub(r'-{3,}', '—', text)

    return text.strip()


def count_tokens(text: str, model: str = "claude-sonnet-4-20250514") -> int:
    
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))




def chunk_text(text: str, max_tokens: int = 600, overlap_tokens: int = 80) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,       # ~4 chars per token as a rough guide
        chunk_overlap=overlap_tokens * 4,
        separators=["\n\n", "\n", ". ", " ", ""],  # Semantic priority order
    )
    chunks = splitter.split_text(text)

    
    enc = tiktoken.get_encoding("cl100k_base")
    validated = []
    for chunk in chunks:
        token_count = len(enc.encode(chunk))
        if token_count <= max_tokens:
            validated.append(chunk)
        else:
           
            words = chunk.split()
            sub = []
            sub_tokens = 0
            for word in words:
                wt = len(enc.encode(word))
                if sub_tokens + wt > max_tokens:
                    validated.append(' '.join(sub))
                    sub, sub_tokens = [], 0
                sub.append(word)
                sub_tokens += wt
            if sub:
                validated.append(' '.join(sub))

    return validated



def enrich_chunk(
    chunk: str,
    source: str = "",
    section: str = "",
    date: str = "",
    index: int = 0,
) -> str:
    header_parts = []
    if source:  header_parts.append(f"SOURCE: {source}")
    if section: header_parts.append(f"SECTION: {section}")
    if date:    header_parts.append(f"DATE: {date}")
    header_parts.append(f"CHUNK: {index + 1}")

    header = "[" + " | ".join(header_parts) + "]"
    return f"{header}\n{chunk}"


def sanitize_for_llm(
    raw_text: str,
    source: str = "",
    section: str = "",
    date: str = "",
    redact_pii_flag: bool = True,
    max_tokens_per_chunk: int = 600,
) -> list[dict]:

    text = structural_clean(raw_text)
    text = remove_noise(text)

    if redact_pii_flag:
        text = redact_pii(text)

    text = compress_text(text)
    chunks = chunk_text(text, max_tokens=max_tokens_per_chunk)

    result = []
    for i, chunk in enumerate(chunks):
        enriched = enrich_chunk(chunk, source, section, date, i)
        result.append({
            "index": i,
            "token_count": count_tokens(enriched),
            "text": enriched,
        })

    return result