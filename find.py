from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import sqlite3


def inspect_metadata_db():
    print("\n" + "="*30 + " [SQL 数据库内容透视] " + "="*30)
    # 确保库名一致
    conn = sqlite3.connect("gemini_metadata.db")
    cursor = conn.cursor()

    try:
        # 查询所有记录
        cursor.execute("SELECT * FROM docs")
        rows = cursor.fetchall()

        if not rows:
            print("[!] 数据库目前是空的。")
        else:
            # 优化间距：文件名给 60 字符，时间给 20 字符
            header = f"{'文件名 (filename)':<60} | {'上传日期 (upload_date)':<20}"
            print(header)
            print("-" * len(header))

            for row in rows:
                # row[0] 是全路径或文件名，row[1] 是动态生成的日期时间
                filename = row[0]
                # 如果路径太长，只显示最后 57 个字符，前面加省略号，保证表格对齐
                display_name = (filename[:570] + '...') if len(filename) > 600 else filename
                print(f"{display_name:<600} | {row[1]:<200}")

        print(f"\n[统计] 总计记录条数: {len(rows)}")

    except Exception as e:
        print(f"读取失败: {e}")
    finally:
        conn.close()
    print("="*82 + "\n")




# 1. 初始化相同的 Embedding 模型以读取向量
os.environ["GOOGLE_API_KEY"] = "AIzaSyAYzlGTiQl0uqN_mvmxfTK86WO7vHk8v0Q"
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. 加载本地 FAISS 索引
vector_db = FAISS.load_local("faiss_gemini_index", embeddings, allow_dangerous_deserialization=True)

# 3. 提取 FAISS 内部存储的所有文本块
print(f"=== [FAISS 向量库内容透视] ===")
# 获取所有文档 ID 及其内容
doc_dict = vector_db.docstore._dict

print(f"检测到总文本块数量: {len(doc_dict)}")
print("-" * 50)

for i, (doc_id, doc) in enumerate(list(doc_dict.items())[:100]): # 仅展示前5个示例
    print(f"【文本块 #{i+1}】")
    print(f"内容摘要: {doc.page_content[:5]}...") # 展示前150个字符
    print(f"关联元数据 (连接SQL的纽带): {doc.metadata}")
    print("-" * 50)

inspect_metadata_db()

