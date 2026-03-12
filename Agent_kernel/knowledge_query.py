"""
知识库查询接口：从已建好的 Chroma 知识库检索（本地持久化），返回完整文本列表。
供后续「知识库节点」调用，与 search_results 类似的契约。
使用本地 models/bge-m3 作为 embedding，与建库一致。
"""
import os
from pathlib import Path
from typing import List

COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
PERSIST_DIR = Path(os.environ.get("KB_PERSIST_DIR", str(Path(__file__).resolve().parent / "chroma_db"))).resolve()
BGE_M3_PATH = Path(__file__).resolve().parent / "models" / "bge-m3"
EMBEDDING_DIM = 1024


def query_knowledge(question: str, top_k: int = 5) -> List[str]:
    """根据问题在本地持久化知识库中检索，返回（当前块 + 下一块）的文本列表。"""
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

    out: List[str] = []
    for node in nodes:
        text = getattr(node, "text", None)
        meta = getattr(node, "metadata", None) or {}
        if text:
            out.append(text)

        # 严格拼接：当前块 + 下一块（同 source_file，chunk_seq+1）
        try:
            source_file = meta.get("source_file")
            chunk_seq = meta.get("chunk_seq")
            if source_file is None or chunk_seq is None:
                continue
            next_filters = MetadataFilters(
                filters=[
                    ExactMatchFilter(key="source_file", value=source_file),
                    ExactMatchFilter(key="chunk_seq", value=int(chunk_seq) + 1),
                ]
            )
            next_retriever = index.as_retriever(similarity_top_k=1, filters=next_filters)
            next_nodes = next_retriever.retrieve(question)
            if next_nodes:
                next_text = getattr(next_nodes[0], "text", None)
                if next_text:
                    out.append(next_text)
        except Exception:
            # 兼容不同版本/后端：若过滤不可用则跳过 next chunk
            continue

    return out


if __name__ == "__main__":
    import sys
    #q = sys.argv[1] if len(sys.argv) > 1 else "How can I use printf function"
    q = sys.argv[1] if len(sys.argv) > 1 else "How can I use the assert function"
    for i, t in enumerate(query_knowledge(q, top_k=3)):
        #snip = (t[:500] + "...") if len(t) > 5000 else t
        snip = t
        print(f"--- [{i+1}] ---\n{snip}\n")
