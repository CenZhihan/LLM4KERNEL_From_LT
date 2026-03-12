"""
仅清空知识库对应的 Milvus collection，不重建索引。
用于「只清空、不重建」的场景。建库请运行 build_knowledge_base.py。
"""
import asyncio
import os

MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
COLLECTION_NAME = os.environ.get("MILVUS_COLLECTION", "ascend_c_knowledge")
EMBEDDING_DIM = 1024


async def _clear_async() -> None:
    from llama_index.vector_stores.milvus import MilvusVectorStore

    vector_store = MilvusVectorStore(
        uri=MILVUS_URI,
        dim=EMBEDDING_DIM,
        collection_name=COLLECTION_NAME,
        overwrite=False,
    )
    if hasattr(vector_store, "clear"):
        vector_store.clear()
        print(f"已清空 collection: {COLLECTION_NAME}")
    else:
        MilvusVectorStore(
            uri=MILVUS_URI,
            dim=EMBEDDING_DIM,
            collection_name=COLLECTION_NAME,
            overwrite=True,
        )
        print(f"已通过 overwrite 清空并重建空 collection: {COLLECTION_NAME}")


def main() -> None:
    asyncio.run(_clear_async())


if __name__ == "__main__":
    main()
