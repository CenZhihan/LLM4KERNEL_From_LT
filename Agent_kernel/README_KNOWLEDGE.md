# 知识库：PDF + Chroma(本地持久化) + LlamaIndex

将 `Knowledge/` 目录下的 PDF（如 Ascend C API Reference）建入本地 Chroma（持久化目录在项目内），供检索使用。  
- **按章节切分**：使用 Unstructured 解析 PDF，按 Title/Header 等元素分组为章节，避免在代码块中间截断。  
- **本地 BGE-M3**：Embedding 使用当前目录下 `models/bge-m3`，无需 OpenAI key。  
建库时默认写入 `chroma_db/`，把整个项目（包含 `chroma_db/`）拷贝到服务器即可直接检索使用。

## 1. 本地持久化目录

默认持久化目录：`Agent_kernel/chroma_db/`（可用环境变量 `KB_PERSIST_DIR` 覆盖）。

该目录是本地数据库文件。若你希望服务器“拷贝项目即可用”，请将该目录一并带上（不一定要提交到 git，也可以打包拷贝）。

## 2. 环境与依赖

- Python 依赖见 `requirements.txt`（含 `llama-index-embeddings-huggingface`、`unstructured` 等）。
- 建库与查询使用**本地 BGE-M3**：请将模型放在 `models/bge-m3`（与 `build_knowledge_base.py` 同目录下），无需 OpenAI key。

## 3. 建库（清空并重建）

首次或需要重建时运行。若要重建（先删除旧库），设置 `KB_REBUILD=1`：

```bash
KB_REBUILD=1 python build_knowledge_base.py
```

- PDF 路径：`Knowledge/` 目录下。
- 切分：按 **章节**（Unstructured 识别的 Title/Header）分组，每章一块，保证代码不截断。
- 可选环境变量：`KB_PERSIST_DIR`、`KB_COLLECTION`、`KB_SKIP_FIRST_PAGES`、`KB_INSERT_BATCH_SIZE`。

## 5. 查询接口

在代码中调用：

```python
from knowledge_query import query_knowledge

texts = query_knowledge("Ascend C 算子如何定义", top_k=5)
# texts: List[str]，每条为检索到的一段完整文本
```

命令行快速测试：

```bash
python knowledge_query.py "你的问题"
```

后续可将 `query_knowledge(...)` 作为「知识库节点」接入 LangGraph Agent（与 `search_results` 类似）。
