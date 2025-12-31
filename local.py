import os
import sqlite3
from datetime import datetime

import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI, \
    GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_classic.chains import RetrievalQA
import gradio as gr

# ================= 1. 配置区 =================
# 在这里填入你的 Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYzlGTiQl0uqN_mvmxfTK86WO7vHk8v0Q"


# ================= 2. 数据处理与语义分块 =================
def process_document(path):
    print(f"[*] 正在解析 PDF: {path}")
    doc = fitz.open(path)
    text = "".join([page.get_text() for page in doc])

    # 使用 Gemini 专用的 Embedding 模型
    # task_type="retrieval_document" 会告诉模型这是为了存入数据库供后续搜索用的
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004",
                                              task_type="retrieval_document")

    print("[*] 正在执行语义分块 (Gemini 会根据语义重心进行智能切分)...")
    # 相比固定字数切分，这能让检索准确率显著提升
    semantic_splitter = SemanticChunker(embeddings,
                                        breakpoint_threshold_type="percentile")
    chunks = semantic_splitter.create_documents([text])
    return chunks, embeddings


# ================= 3. 混合存储 (SQL + FAISS) =================
def setup_databases(chunks, embeddings, filename):
    print(f"[*] 正在为 {os.path.basename(filename)} 构建混合数据库...")

    # 1. 动态获取当前上传日期
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 为每个块标记来源，实现简历中的“元数据过滤”
    for chunk in chunks:
        chunk.metadata = {"source": filename}

    # 2. 构建/更新向量库 (FAISS)
    # 如果本地已有索引，建议使用 merge 或直接覆盖。这里我们采用保存逻辑：
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_gemini_index")

    # 3. 构建 SQL 元数据表 (SQLite)
    conn = sqlite3.connect("gemini_metadata.db")
    c = conn.cursor()

    # 创建表结构
    c.execute(
        "CREATE TABLE IF NOT EXISTS docs (filename TEXT, upload_date TEXT)")

    # 【核心完善】：先检查 SQL 中是否已存在该文件，避免重复插入导致搜索噪音
    c.execute("SELECT filename FROM docs WHERE filename = ?", (filename,))
    if c.fetchone() is None:
        c.execute("INSERT INTO docs VALUES (?, ?)", (filename, current_date))
        print(
            f"[+] SQL 登记完成: {os.path.basename(filename)} (日期: {current_date})")
    else:
        # 如果文件已存在，则更新其上传日期
        c.execute("UPDATE docs SET upload_date = ? WHERE filename = ?",
                  (current_date, filename))
        print(f"[*] SQL 已更新现有记录的日期: {current_date}")

    conn.commit()
    conn.close()


# ================= 4. 检索与问答引擎 =================
def run_gemini_qa(query, vector_db, filename):
    # 核心逻辑：直接使用外部传入的 vector_db，省去重复加载的时间
    retriever = vector_db.as_retriever(
        search_kwargs={'filter': {'source': filename}, 'k': 4}
    )

    # 修正模型名称：确保使用你 API key 支持的有效名称
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                           retriever=retriever)

    print(f"\n[问题]: {query}")
    response = qa_chain.invoke(query)
    print(f"[Gemini 回答]: {response['result']}")


def should_process_file(filename):
    """检查 SQL 库里是否已经索引过这个文件"""
    conn = sqlite3.connect("gemini_metadata.db")
    cursor = conn.cursor()

    # 查找文件名是否存在
    cursor.execute("SELECT filename FROM docs WHERE filename = ?", (filename,))
    result = cursor.fetchone()
    conn.close()

    if result:
        print(
            f"[*] 文件 {os.path.basename(filename)} 已在索引中，跳过解析直接进入问答。")
        return False
    return True


if __name__ == "__main__":
    PDF_PATH = (r"C:\Users\金培杰\OneDrive - University of "
                r"Toronto\桌面\individual_contribution.pdf")
    if not os.path.exists(PDF_PATH):
        print(f"请准备好名为 {PDF_PATH} 的文件")
    else:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            task_type="retrieval_document")
        if should_process_file(PDF_PATH):
            # 执行原有的解析、语义分块和 setup_databases
            chunks, emb = process_document(PDF_PATH)
            setup_databases(chunks, emb, PDF_PATH)
            vector_db = FAISS.load_local("faiss_gemini_index", embeddings,
                                         allow_dangerous_deserialization=True)
        else:
            # 直接加载现有的 FAISS 索引进行问答
            print("有的有的")
            vector_db = FAISS.load_local("faiss_gemini_index", embeddings,
                                         allow_dangerous_deserialization=True)
        run_gemini_qa("请总结这份文档的核心观点", vector_db, PDF_PATH)



