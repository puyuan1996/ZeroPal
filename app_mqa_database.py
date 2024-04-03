import os
import sqlite3
import threading

import gradio as gr
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

from RAG.analyze_conversation_history import analyze_conversation_history
from rag_demo import load_and_split_document, create_vector_store, setup_rag_chain, execute_query

# 环境设置
load_dotenv()  # 加载环境变量
QUESTION_LANG = os.getenv("QUESTION_LANG")  # 从环境变量获取 QUESTION_LANG
assert QUESTION_LANG in ['cn', 'en'], QUESTION_LANG

if QUESTION_LANG == "cn":
    title = "ZeroPal"
    title_markdown = """
    <div align="center">
        <img src="https://raw.githubusercontent.com/puyuan1996/RAG/main/assets/banner.svg" width="80%" height="20%" alt="Banner Image">
    </div>
    
    📢 **操作说明**：请在下方的“问题”框中输入关于 LightZero 的问题，并点击“提交”按钮。右侧的“回答”框将展示 RAG 模型提供的答案。
    您可以在问答框下方查看当前“对话历史”，点击“清除上下文”按钮可清空历史记录。在“对话历史”框下方，您将找到相关参考文档，其中相关文段将以黄色高亮显示。
    如果您喜欢这个项目，请在 GitHub [LightZero RAG Demo](https://github.com/puyuan1996/RAG) 上给我们点赞！✨ 您的支持是我们持续更新的动力。
    
    <div align="center">
        <strong>注意：算法模型输出可能包含一定的随机性。结果不代表开发者和相关 AI 服务的态度和意见。本项目开发者不对结果作出任何保证，仅供参考之用。使用该服务即代表同意后文所述的使用条款。</strong>
    </div>
    """
    tos_markdown = """
    ### 使用条款
    
    使用本服务的玩家需同意以下条款：
    
    - 本服务为探索性研究的预览版，仅供非商业用途。
    - 服务不得用于任何非法、有害、暴力、种族主义或其他令人反感的目的。
    - 服务提供有限的安全措施，并可能生成令人反感的内容。
    - 如果您对服务体验不满，请通过 opendilab@pjlab.org.cn 与我们联系！我们承诺修复问题并不断改进项目。
    - 为了获得最佳体验，请使用台式电脑，因为移动设备可能会影响视觉效果。
    
    **版权所有 © 2024 OpenDILab。保留所有权利。**
    """

# 路径变量,方便之后的文件使用
file_path = './documents/LightZero_README_zh.md'

# 加载原始Markdown文档
loader = TextLoader(file_path)
orig_documents = loader.load()

# 存储对话历史
conversation_history = {}

# 创建线程局部数据对象
threadLocal = threading.local()


def get_db_connection():
    """
    返回当前线程的数据库连接
    """
    conn = getattr(threadLocal, 'conn', None)
    if conn is None:
        # 连接到SQLite数据库
        conn = sqlite3.connect('database/conversation_history.db')
        c = conn.cursor()
        # Drop the existing 'history' table if it exists
        # c.execute('DROP TABLE IF EXISTS history')
        # 创建存储对话历史的表
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id TEXT NOT NULL,
                     user_input TEXT NOT NULL,
                     assistant_output TEXT NOT NULL,
                     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        threadLocal.conn = conn
    return conn


def get_db_cursor():
    """
    返回当前线程的数据库游标
    """
    conn = get_db_connection()
    c = getattr(threadLocal, 'cursor', None)
    if c is None:
        c = conn.cursor()
        threadLocal.cursor = c
    return c


# 程序结束时清理数据库连接
def close_db_connection():
    conn = getattr(threadLocal, 'conn', None)
    if conn is not None:
        conn.close()
        setattr(threadLocal, 'conn', None)

    c = getattr(threadLocal, 'cursor', None)
    if c is not None:
        c.close()
        setattr(threadLocal, 'cursor', None)


def rag_answer(question, temperature, k, user_id):
    """
    处理用户问题并返回答案和高亮显示的上下文

    :param question: 用户输入的问题
    :param temperature: 生成答案时使用的温度参数
    :param k: 检索到的文档块数量
    :param user_id: 用户ID
    :return: 模型生成的答案和高亮显示上下文的Markdown文本
    """
    try:
        chunks = load_and_split_document(file_path, chunk_size=5000, chunk_overlap=500)
        retriever = create_vector_store(chunks, model='OpenAI', k=k)
        rag_chain = setup_rag_chain(model_name='kimi', temperature=temperature)

        if user_id not in conversation_history:
            conversation_history[user_id] = []

        conversation_history[user_id].append((f"User[{user_id}]", question))

        history_str = "\n".join([f"{role}: {text}" for role, text in conversation_history[user_id]])

        retrieved_documents, answer = execute_query(retriever, rag_chain, history_str, model_name='kimi',
                                                    temperature=temperature)

        ############################
        # 获取当前线程的数据库连接和游标
        ############################
        conn = get_db_connection()
        c = get_db_cursor()

        # 分析对话历史
        # analyze_conversation_history()
        # 获取总的对话记录数
        c.execute("SELECT COUNT(*) FROM history")
        total_records = c.fetchone()[0]
        print(f"总对话记录数: {total_records}")

        # 将问题和回答存储到数据库
        c.execute("INSERT INTO history (user_id, user_input, assistant_output) VALUES (?, ?, ?)",
                  (user_id, question, answer))
        conn.commit()

        # 在文档中高亮显示上下文
        context = [retrieved_documents[i].page_content for i in range(len(retrieved_documents))]
        highlighted_document = orig_documents[0].page_content
        for i in range(len(context)):
            highlighted_document = highlighted_document.replace(context[i], f"<mark>{context[i]}</mark>")

        conversation_history[user_id].append(("Assistant", answer))

        full_history = "\n".join([f"{role}: {text}" for role, text in conversation_history[user_id]])
    except Exception as e:
        print(f"An error occurred: {e}")
        return "处理您的问题时出现错误,请稍后再试。", "", ""
    finally:
        # 不再在这里关闭游标和连接
        pass

    return answer, highlighted_document, full_history


def clear_context(user_id):
    """
    清除对话历史
    """
    if user_id in conversation_history:
        conversation_history[user_id] = []
    return "", "", ""


if __name__ == "__main__":
    with gr.Blocks(title=title, theme='ParityError/Interstellar') as zero_pal:
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column():
                user_id = gr.Textbox(
                    placeholder="请输入您的真实姓名或昵称作为用户ID",
                    label="用户ID")
                inputs = gr.Textbox(
                    placeholder="请您在这里输入任何关于 LightZero 的问题。",
                    label="问题")
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.01, step=0.01, label="温度参数")
                k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="检索到的文档块数量")
                with gr.Row():
                    gr_submit = gr.Button('提交')
                    gr_clear = gr.Button('清除上下文')

            outputs_answer = gr.Textbox(placeholder="当你点击提交按钮后,这里会显示 RAG 模型给出的回答。",
                                        label="回答")
        outputs_history = gr.Textbox(label="对话历史")
        with gr.Row():
            outputs_context = gr.Markdown(label="参考的文档(检索得到的相关文段用高亮显示)")
        gr_clear.click(clear_context, inputs=user_id, outputs=[outputs_context, outputs_history])
        gr_submit.click(
            rag_answer,
            inputs=[inputs, temperature, k, user_id],
            outputs=[outputs_answer, outputs_context, outputs_history],
        )
        gr.Markdown(tos_markdown)

    concurrency = int(os.environ.get('CONCURRENCY', os.cpu_count()))
    favicon_path = os.path.join(os.path.dirname(__file__), 'assets', 'avatar.png')
    zero_pal.queue().launch(max_threads=concurrency, favicon_path=favicon_path, share=True)

    # 在合适的地方，例如程序退出时，调用close_db_connection函数
    close_db_connection()
