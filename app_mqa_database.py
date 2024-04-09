import collections
import os
import sqlite3
import threading

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer, util

from rag_demo import load_and_split_document, create_vector_store, setup_rag_chain, execute_query, get_retriever

# 环境设置
load_dotenv()  # 加载环境变量
QUESTION_LANG = os.getenv("QUESTION_LANG")  # 从环境变量获取 QUESTION_LANG
assert QUESTION_LANG in ['cn', 'en'], QUESTION_LANG

if QUESTION_LANG == "cn":
    title = "ZeroPal"
    title_markdown = """
    <div align="center">
        <img src="https://raw.githubusercontent.com/puyuan1996/ZeroPal/main/assets/banner.svg" width="80%" height="20%" alt="Banner Image">
    </div>

    📢 **操作说明**：请在下方的"问题"框中输入关于 LightZero 的问题，并点击"提交"按钮。右侧的"回答"框将展示 RAG 模型提供的答案。
    您可以在问答框下方查看当前"对话历史"，点击"清除对话历史"按钮可清空历史记录。在"对话历史"框下方，您将找到相关参考文档，其中相关文段将以黄色高亮显示。
    如果您喜欢这个项目，请在 GitHub [LightZero RAG Demo](https://github.com/puyuan1996/ZeroPal) 上给我们点赞！✨ 您的支持是我们持续更新的动力。注意：算法模型输出可能包含一定的随机性。结果不代表开发者和相关 AI 服务的态度和意见。本项目开发者不对结果作出任何保证，仅供参考之用。使用该服务即代表同意后文所述的使用条款。

    📢 **Instructions**: Please enter your questions about LightZero in the "Question" box below and click the "Submit" button. The "Answer" box on the right will display the answers provided by the RAG model.
    Below the Q&A box, you can view the current "Conversation History". Clicking the "Clear Conversation History" button will erase the history records. Below the "Conversation History" box, you'll find relevant reference documents, with the pertinent sections highlighted in yellow.
    If you like this project, please give us a thumbs up on GitHub at [LightZero RAG Demo](https://github.com/puyuan1996/ZeroPal)! ✨ Your support motivates us to keep updating.
    Note: The output from the algorithm model may contain a degree of randomness. The results do not represent the attitudes and opinions of the developers and related AI services. The developers of this project make no guarantees about the results, which are for reference only. Use of this service indicates agreement with the terms of use described later in the text.
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
        # 创建存储对话历史的表
        conn = sqlite3.connect('database/conversation_history.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS history
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_id TEXT NOT NULL,
                     user_input TEXT NOT NULL,
                     user_input_embedding BLOB NOT NULL,
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


chunks = load_and_split_document(file_path, chunk_size=5000, chunk_overlap=500)
vectorstore = create_vector_store(chunks, model='OpenAI')

# 加载预训练的SBERT模型
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 定义余弦相似度阈值
cosine_threshold = 0.96  # 为了提高检索的准确性，将余弦相似度阈值调高

# 设置LRU缓存的大小
CACHE_SIZE = 1000

# 创建历史问题的缓存
conversation_history_cache = collections.OrderedDict()


# def rag_answer(question, temperature=0.01, k=5, user_id='user'):
def rag_answer(question, k=5, user_id='user'):
    """
    处理用户问题并返回答案和高亮显示的上下文

    :param question: 用户输入的问题
    :param temperature: 生成答案时使用的温度参数
    :param k: 检索到的文档块数量
    :param user_id: 用户ID
    :return: 模型生成的答案和高亮显示上下文的Markdown文本
    """
    temperature = 0.01  # TODO: 使用固定的温度参数

    try:
        # 获取当前线程的数据库连接和游标
        conn = get_db_connection()
        c = get_db_cursor()

        question_embedding = sbert_model.encode(question)
        question_embedding_bytes = question_embedding.tobytes()  # 将numpy数组转换为字节串

        # 从数据库中获取所有用户的对话历史
        c.execute("SELECT user_input, user_input_embedding, assistant_output FROM history")
        all_history = c.fetchall()
        # 初始化最高的余弦相似度和对应的答案
        max_cosine_score = 0
        best_answer = ""
        # 在历史问题的缓存中查找相似问题
        for history_question_bytes, (history_question, history_answer) in conversation_history_cache.items():
            history_question_embedding_numpy = np.frombuffer(history_question_bytes, dtype=np.float32)
            cosine_score = util.cos_sim(question_embedding, history_question_embedding_numpy).item()
            # print(f"检索到历史问题: {history_question}")
            # print(f"当前问题与历史问题的余弦相似度: {cosine_score}")
            if cosine_score > cosine_threshold and cosine_score > max_cosine_score:
                max_cosine_score = cosine_score
                best_answer = history_answer

        if user_id not in conversation_history:
            conversation_history[user_id] = []

        conversation_history[user_id].append((f"User[{user_id}]", question))
        # 如果余弦相似度高于阈值,则更新最佳答案
        if max_cosine_score > cosine_threshold:
            print('=' * 20)
            print(f"找到了足够相似的历史问题,直接返回对应的答案。余弦相似度为: {max_cosine_score}")
            answer = best_answer
        else:
            retriever = get_retriever(vectorstore, k)
            rag_chain = setup_rag_chain(model_name='kimi', temperature=temperature)
            history_str = "\n".join([f"{role}: {text}" for role, text in conversation_history[user_id]])
            history_question = [history_str, question]
            retrieved_documents, answer = execute_query(retriever, rag_chain, history_question, model_name='kimi',
                                                        temperature=temperature)

        # 获取总的对话记录数
        c.execute("SELECT COUNT(*) FROM history")
        total_records = c.fetchone()[0]
        print(f"总对话记录数: {total_records}")

        # 将问题和回答存储到数据库
        c.execute(
            "INSERT INTO history (user_id, user_input, user_input_embedding, assistant_output) VALUES (?, ?, ?, ?)",
            (user_id, question, question_embedding_bytes, answer))
        conn.commit()

        # 将新问题和答案添加到历史问题的缓存中
        conversation_history_cache[question_embedding_bytes] = (question, answer)
        # 如果缓存大小超过限制,则淘汰最近最少使用的问题
        if len(conversation_history_cache) > CACHE_SIZE:
            conversation_history_cache.popitem(last=False)

        if max_cosine_score > cosine_threshold:
            highlighted_document = ""
        else:
            # 在文档中高亮显示上下文
            context = [retrieved_documents[i].page_content for i in range(len(retrieved_documents))]
            highlighted_document = orig_documents[0].page_content
            for i in range(len(context)):
                highlighted_document = highlighted_document.replace(context[i], f"<mark>{context[i]}</mark>")

        conversation_history[user_id].append(("Assistant", answer))
        full_history = "\n".join([f"{role}: {text}" for role, text in conversation_history[user_id]])

    except Exception as e:
        print(f"An error occurred: {e}")
        return f"处理您的问题时出现错误,请稍后再试。错误内容为：{e}", "", ""
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
                    placeholder="请输入您的真实姓名或昵称作为用户ID(Please enter your real name or nickname as the user ID.)",
                    label="用户ID(Username)")
                inputs = gr.Textbox(
                    placeholder="请您在这里输入任何关于 LightZero 的问题。(Please enter any questions about LightZero here.)",
                    label="问题(Question)")
                # temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.01, step=0.01, label="温度参数")
                k = gr.Slider(minimum=1, maximum=7, value=3, step=1,
                              label="检索到的相关文档块的数量(The number of relevant document blocks retrieved.)")  # readme总长度为35000左右，文段块长度为5000，因此最大值为35000/5000=7
                with gr.Row():
                    gr_submit = gr.Button('提交(Submit)')
                    gr_clear = gr.Button('清除对话历史(Clear Context)')

            outputs_answer = gr.Textbox(
                placeholder="当你点击提交按钮后,这里会显示 RAG 模型给出的回答。（After you click the submit button, the answer given by the RAG model will be displayed here.）",
                label="回答(Answer)")
        outputs_history = gr.Textbox(label="对话历史(Conversation History)")
        with gr.Row():
            outputs_context = gr.Markdown(
                label="参考的文档(检索得到的相关文段用高亮显示) Referenced documents (the relevant excerpts retrieved are highlighted).")
        gr_clear.click(clear_context, inputs=user_id, outputs=[outputs_context, outputs_history])
        gr_submit.click(
            rag_answer,
            # inputs=[inputs, temperature, k, user_id],
            inputs=[inputs, k, user_id],
            outputs=[outputs_answer, outputs_context, outputs_history],
        )
        gr.Markdown(tos_markdown)

    concurrency = int(os.environ.get('CONCURRENCY', os.cpu_count()))
    favicon_path = os.path.join(os.path.dirname(__file__), 'assets', 'avatar.png')
    zero_pal.queue().launch(max_threads=concurrency, favicon_path=favicon_path, share=True)

    # 在合适的地方,例如程序退出时,调用close_db_connection函数
    close_db_connection()
