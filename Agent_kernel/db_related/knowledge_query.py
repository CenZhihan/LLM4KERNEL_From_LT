"""
知识库查询接口：从已建好的 Chroma 知识库检索（本地持久化），返回完整文本列表。
供后续「知识库节点」调用，与 search_results 类似的契约。
使用本地 models/bge-m3 作为 embedding，与建库一致。
"""
import os
from pathlib import Path
from typing import List

COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
PERSIST_DIR = Path(
    os.environ.get(
        "KB_PERSIST_DIR",
        str(Path(__file__).resolve().parent.parent / "chroma_db"),
    )
).resolve()
BGE_M3_PATH = Path(__file__).resolve().parent.parent / "models" / "bge-m3"
EMBEDDING_DIM = 1024


def query_knowledge(question: str, top_k: int = 5) -> List[str]:
    """根据问题在本地持久化知识库中检索，返回「匹配到的 chunk_seq 下所有物理块」的文本列表。

    注意：同一 chunk_seq 下可能有多个物理 chunk（存在 overlap），这里会为每个物理块加清晰分隔。
    """
    from llama_index.core import VectorStoreIndex
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = HuggingFaceEmbedding(
        model_name=str(BGE_M3_PATH),
        trust_remote_code=True,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    def _collect_seq_docs(source_file: str, seq_int: int) -> List[str]:
        """在同一 source_file 下，收集指定 chunk_seq 的所有文本块，按 Chroma 返回顺序拼接。"""

        def _get_docs(val):
            where = {
                "$and": [
                    {"source_file": source_file},
                    {"chunk_seq": val},
                ]
            }
            raw = chroma_collection.get(where=where)
            return raw.get("documents") or []

        docs = _get_docs(seq_int)
        if not docs:
            docs = _get_docs(str(seq_int))
        return docs

    def _format_seq_docs(seq_int: int, docs: List[str]) -> str:
        """为同一 seq 下的多个物理 chunk 做清晰分隔，避免 overlap 混在一起难以阅读。"""
        if not docs:
            return ""
        parts: List[str] = []
        for idx, d in enumerate(docs, start=1):
            header = f"----- [seq={seq_int}, part={idx}] -----"
            parts.append(header)
            parts.append(d.strip())
        return "\n\n".join(parts)

    out: List[str] = []
    for rank, node in enumerate(nodes, start=1):
        meta = getattr(node, "metadata", None) or {}

        # 当前块：按逻辑 seq 收集该章节下所有物理块
        try:
            source_file = meta.get("source_file")
            chunk_seq = meta.get("chunk_seq")
            if source_file is not None and chunk_seq is not None:
                cur_seq_int = int(chunk_seq)
                cur_docs = _collect_seq_docs(str(source_file), cur_seq_int)
            else:
                cur_docs = []
        except Exception:
            cur_docs = []

        text = _format_seq_docs(cur_seq_int, cur_docs) if cur_docs else getattr(node, "text", None)
        if not text:
            continue

        # 每个结果：只返回当前 seq 下的所有块
        entry_lines = [
            f"=== TOP {rank} ===",
            "[current]",
            text,
        ]
        entry = "\n".join(entry_lines)
        out.append(entry)

    return out


if __name__ == "__main__":
    import sys
    #q = sys.argv[1] if len(sys.argv) > 1 else "How can I use printf function"
    q = sys.argv[1] if len(sys.argv) > 1 else "How can I use the assert function"
    for t in query_knowledge(q, top_k=3):
        snip = t
        print(f"{snip}\n")
