"""
知识库建库脚本：从 Knowledge/ 目录加载 PDF，按章节（Title/Header）切分，
写入本地 Chroma（持久化目录在项目内，拷贝项目即可用）。
支持重建：删除本地持久化目录后重新建库。
Embedding 使用当前目录下 models/bge-m3 本地模型。
运行前无需启动外部服务。
"""

import os
from pathlib import Path
import shutil

# 配置：可从环境变量覆盖
COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
# 本文件移动到 db_related/ 后，默认持久化目录仍放在项目根目录下的 chroma_db
PERSIST_DIR = Path(
    os.environ.get(
        "KB_PERSIST_DIR",
        str(Path(__file__).resolve().parent.parent / "chroma_db"),
    )
).resolve()
# 重建：1 表示先删掉持久化目录
# 这里如果不传入参数，则默认重建
REBUILD = os.environ.get("KB_REBUILD", "1") in ("1", "true", "True", "yes", "YES")
# Knowledge 目录与 models 目录仍位于项目根目录
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "Knowledge"
# 本地 BGE-M3 路径（hidden_size=1024）
BGE_M3_PATH = Path(__file__).resolve().parent.parent / "models" / "bge-m3"
EMBEDDING_DIM = 1024

# 忽略 PDF 前 N 页（本 PDF 前 22 页为目录）
SKIP_FIRST_PAGES = int(os.environ.get("KB_SKIP_FIRST_PAGES", "22"))

# 视为「新章节开始」的 Unstructured 元素类型，用于按章节切分
SECTION_HEADER_CATEGORIES = ("Title", "Header", "Section Header", "Page Header")


def _iter_pdf_sections_pypdf(pdf_path: Path):
    """备选：用 pypdf 按页流式解析，再按章节行切分（不依赖 pi_heif，低内存）。"""
    from llama_index.core import Document
    from pypdf import PdfReader
    import re

    # 行首章节模式：1 / 1.2 / 1.2.3 / Chapter / 第X章
    header_re = re.compile(r"^(?:\d+(?:\.\d+)*\s+.+|Chapter\s+\d+.*|第[一二三四五六七八九十\d]+章.*)$")

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    current_lines = []
    saw_any_text = False
    chunk_seq = 0
    source_file = pdf_path.name

    for page_idx, page in enumerate(reader.pages, start=1):
        if page_idx <= SKIP_FIRST_PAGES:
            continue
        text = page.extract_text() or ""
        if text.strip():
            saw_any_text = True
        # 每 10 页打印一次进度（避免刷屏）
        if page_idx == 1 or page_idx % 10 == 0 or page_idx == total_pages:
            print(f"    [pypdf] pages {page_idx}/{total_pages}")

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if header_re.match(line):
                if current_lines:
                    yield Document(
                        text="\n".join(current_lines).strip(),
                        metadata={"source_file": source_file, "chunk_seq": chunk_seq},
                    )
                    chunk_seq += 1
                    current_lines = []
                current_lines.append(line)
            else:
                current_lines.append(line)

    if current_lines:
        yield Document(
            text="\n".join(current_lines).strip(),
            metadata={"source_file": source_file, "chunk_seq": chunk_seq},
        )
    elif not saw_any_text:
        yield Document(text="(未提取到文本)", metadata={"source_file": source_file, "chunk_seq": chunk_seq})


def _iter_pdf_sections(pdf_path: Path):
    """优先用 unstructured 按 Title/Header 分组；若无 pi_heif 则退回 pypdf 流式切分。"""
    from llama_index.core import Document

    try:
        from unstructured.partition.pdf import partition_pdf
    except ModuleNotFoundError as e:
        if getattr(e, "name", None) == "pi_heif" or "pi_heif" in str(e):
            yield from _iter_pdf_sections_pypdf(pdf_path)
            return
        raise

    elements = partition_pdf(filename=str(pdf_path), strategy="auto")
    sections = []
    current_lines = []
    chunk_seq = 0
    source_file = pdf_path.name

    for el in elements:
        # 尝试跳过目录页（依赖 unstructured 元素元数据里的 page_number）
        page_num = None
        meta = getattr(el, "metadata", None)
        if meta is not None:
            page_num = getattr(meta, "page_number", None)
        if isinstance(page_num, int) and page_num <= SKIP_FIRST_PAGES:
            continue
        category = getattr(el, "category", None) or ""
        text = (getattr(el, "text", None) or "").strip()
        if not text:
            continue
        if category in SECTION_HEADER_CATEGORIES:
            if current_lines:
                sections.append("\n\n".join(current_lines))
                current_lines = []
            current_lines.append(text)
        else:
            current_lines.append(text)

    if current_lines:
        sections.append("\n\n".join(current_lines))

    for s in sections:
        yield Document(text=s, metadata={"source_file": source_file, "chunk_seq": chunk_seq})
        chunk_seq += 1


def _run_build() -> None:
    """构建/重建 Chroma 知识库（本地持久化）。"""
    from llama_index.core import VectorStoreIndex, StorageContext
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import chromadb

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(f"知识库目录不存在: {KNOWLEDGE_DIR}")
    if not BGE_M3_PATH.exists():
        raise FileNotFoundError(f"本地 BGE-M3 不存在: {BGE_M3_PATH}，请将模型放在 models/bge-m3")

    pdf_files = list(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"未在 {KNOWLEDGE_DIR} 下找到 PDF 文件")

    embed_model = HuggingFaceEmbedding(
        model_name=str(BGE_M3_PATH),
        trust_remote_code=True,
    )

    if REBUILD and PERSIST_DIR.exists():
        print(f"[rebuild] 删除持久化目录: {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR)

    # Chroma 本地持久化（将 chroma_db 目录随项目打包迁移即可用）
    chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 关键：流式/分批写入，避免把所有章节一次性放进内存
    # 先用空 vector_store 构建 index，再逐批插入 documents
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    batch_size = int(os.environ.get("KB_INSERT_BATCH_SIZE", "2"))
    total_inserted = 0

    for pdf_path in pdf_files:
        print(f"正在按章节解析 PDF: {pdf_path.name} ...")
        print(f"  开始流式写入（batch_size={batch_size}），进度将显示已写入章节数...")
        batch = []
        for doc in _iter_pdf_sections(pdf_path):
            batch.append(doc)
            if len(batch) >= batch_size:
                if hasattr(index, "insert_documents"):
                    index.insert_documents(batch)  # type: ignore[attr-defined]
                    total_inserted += len(batch)
                else:
                    for d in batch:
                        index.insert(d)  # type: ignore[attr-defined]
                        total_inserted += 1
                if total_inserted % 50 == 0:
                    print(f"    已写入章节块: {total_inserted}")
                batch.clear()

        # flush remainder
        if batch:
            if hasattr(index, "insert_documents"):
                index.insert_documents(batch)  # type: ignore[attr-defined]
                total_inserted += len(batch)
            else:
                for d in batch:
                    index.insert(d)  # type: ignore[attr-defined]
                    total_inserted += 1
            batch.clear()
        print(f"  完成该 PDF，累计写入章节块: {total_inserted}")

    if total_inserted <= 0:
        raise RuntimeError("未写入任何章节内容，请检查 PDF 解析结果。")

    print(f"知识库构建完成，共写入 {total_inserted} 个章节块。")


def main() -> None:
    _run_build()


if __name__ == "__main__":
    main()
