"""
简单脚本：直接从本地 Chroma 持久化库中读取文档，
按 (source_file, chunk_seq, id) 排序后，输出前 N 个 chunk，
或从指定序号开始输出后 N 个 chunk（如 --start 1500 输出第 1500～1504 个）。
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import chromadb


COLLECTION_NAME = os.environ.get("KB_COLLECTION", "ascend_c_knowledge")
PERSIST_DIR = Path(
    os.environ.get(
        "KB_PERSIST_DIR",
        str(Path(__file__).resolve().parent.parent / "chroma_db"),
    )
).resolve()


def load_all_chunks(limit: Optional[int] = None) -> List[Tuple[Dict[str, Any], str]]:
    """
    从 Chroma collection 中取出所有（或前 limit 个）文档，返回 [(metadata, document), ...]。
    """
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 直接用 get 取出全部文档；如果数据量特别大，可传入 limit 调小
    raw = collection.get(limit=limit)
    metadatas = raw.get("metadatas") or []
    documents = raw.get("documents") or []

    out: List[Tuple[Dict[str, Any], str]] = []
    for meta, doc in zip(metadatas, documents):
        if doc is None:
            continue
        meta = meta or {}
        out.append((meta, doc))
    return out


def sort_chunks(
    chunks: List[Tuple[Dict[str, Any], str]]
) -> List[Tuple[Dict[str, Any], str]]:
    """
    按 source_file / chunk_seq / 内部顺序 排序。
    """

    def _key(item: Tuple[Dict[str, Any], str]):
        meta, _ = item
        source_file = str(meta.get("source_file") or "")
        # chunk_seq 可能是字符串；取不到时给个很大的数，排在后面
        try:
            chunk_seq = int(meta.get("chunk_seq"))
        except Exception:
            chunk_seq = 10**9
        return (source_file, chunk_seq)

    return sorted(chunks, key=_key)


def pretty_print_chunks(
    chunks: List[Tuple[Dict[str, Any], str]],
    top_k: int = 5,
    display_start: Optional[int] = None,
    total_count: Optional[int] = None,
) -> None:
    """
    漂亮地打印 chunk。若指定 display_start 与 total_count，则显示全局序号（如 1500/1800）。
    """
    if not chunks:
        print("（没有从 Chroma 中读到任何文档）")
        return

    top_k = min(top_k, len(chunks))
    total = total_count if total_count is not None else top_k
    for idx, (meta, doc) in enumerate(chunks[:top_k], start=1):
        global_num = (display_start + idx - 1) if display_start is not None else idx
        header_line = f"===== CHUNK {global_num}/{total} ====="
        source_file = meta.get("source_file", "N/A")
        chunk_seq = meta.get("chunk_seq", "N/A")
        extra_meta = {
            k: v for k, v in meta.items() if k not in {"source_file", "chunk_seq"}
        }

        print(header_line)
        print(f"source_file: {source_file}")
        print(f"chunk_seq  : {chunk_seq}")
        if extra_meta:
            print(f"metadata   : {extra_meta}")
        print("-" * len(header_line))
        print(doc.strip())
        print("=" * len(header_line))
        print()  # 空行分隔


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="从本地 Chroma 知识库中按顺序打印 chunk（前 N 个或从某序号开始的 N 个）。"
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=5,
        help="要输出的 chunk 数量（默认 5）",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=None,
        metavar="N",
        help="从第 N 个 chunk 开始输出（1-based），如 1500 则输出第 1500,1501,...,1504 共 5 个",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="最多从 Chroma 读取多少条数据（默认全部）",
    )

    args = parser.parse_args()

    chunks = load_all_chunks(limit=args.limit)
    sorted_chunks = sort_chunks(chunks)
    total_count = len(sorted_chunks)

    if args.start is not None:
        # 1-based → 0-based，取 [start-1, start-1+top_k)
        start_idx = max(0, args.start - 1)
        end_idx = min(start_idx + args.top_k, total_count)
        slice_chunks = sorted_chunks[start_idx:end_idx]
        if not slice_chunks:
            print(f"（起始序号 {args.start} 超出范围，共 {total_count} 个 chunk）")
        else:
            pretty_print_chunks(
                slice_chunks,
                top_k=args.top_k,
                display_start=args.start,
                total_count=total_count,
            )
    else:
        pretty_print_chunks(sorted_chunks, top_k=args.top_k)


if __name__ == "__main__":
    main()

