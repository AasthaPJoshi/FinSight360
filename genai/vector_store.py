"""
ChromaDB vector store management.
Uses OpenAI text-embedding-3-small (best cost/performance ratio).
Persistent local storage — survives restarts.
"""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from config.settings import settings
from utils.logger import get_logger

log = get_logger("vector_store")

COLLECTION_NAME = "finsight360_filings"


class FilingVectorStore:
    """
    Manages ChromaDB collection for SEC filing chunks.

    ChromaDB chosen over Pinecone/Weaviate:
    - Fully local (no cloud cost, no data privacy concerns)
    - Persistent storage (survives restarts)
    - Native LangChain integration
    """

    def __init__(self):
        chroma_dir = Path(settings.chroma_persist_dir)
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_dir = str(chroma_dir)
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.vectorstore: Chroma | None = None

    def get_or_create(self) -> Chroma:
        """Get existing ChromaDB collection or create new one."""
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self._chroma_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        count = self.vectorstore._collection.count()
        log.info(
            "vectorstore_ready",
            collection=COLLECTION_NAME,
            existing_docs=count,
            path=self._chroma_dir,
        )
        return self.vectorstore

    def add_documents(
        self, documents: list[Document], batch_size: int = 100
    ) -> int:
        """Add documents to ChromaDB in batches, skipping already-indexed sources."""
        if not self.vectorstore:
            self.get_or_create()

        if not documents:
            return 0

        existing_sources = self._get_indexed_sources()
        new_docs = [
            d
            for d in documents
            if d.metadata.get("source") not in existing_sources
        ]

        if not new_docs:
            log.info("all_docs_already_indexed", skipped=len(documents))
            return 0

        added = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            self.vectorstore.add_documents(batch)
            added += len(batch)
            log.info(
                "batch_indexed",
                batch=i // batch_size + 1,
                added=len(batch),
                total=added,
            )

        log.info(
            "documents_indexed", new=added, skipped=len(documents) - added
        )
        return added

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter_cik: str | None = None,
        filter_section: str | None = None,
    ) -> list[Document]:
        """Semantic search with optional filters by CIK or section type."""
        if not self.vectorstore:
            self.get_or_create()

        where_filter: dict = {}
        if filter_cik:
            where_filter["cik"] = filter_cik
        if filter_section:
            where_filter["section_key"] = filter_section

        results = self.vectorstore.similarity_search(
            query, k=k, filter=where_filter if where_filter else None
        )
        log.info(
            "similarity_search",
            query=query[:50],
            results=len(results),
            filter=where_filter,
        )
        return results

    def get_collection_stats(self) -> dict:
        if not self.vectorstore:
            self.get_or_create()
        count = self.vectorstore._collection.count()
        sources = self._get_indexed_sources()
        return {
            "total_chunks": count,
            "unique_sources": len(sources),
            "collection_name": COLLECTION_NAME,
            "persist_dir": self._chroma_dir,
        }

    def _get_indexed_sources(self) -> set[str]:
        try:
            results = self.vectorstore._collection.get(include=["metadatas"])
            return {m.get("source", "") for m in results.get("metadatas", [])}
        except Exception:
            return set()
