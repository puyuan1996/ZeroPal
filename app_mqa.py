import os
import gradio as gr
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from rag_demo import load_and_split_document, create_vector_store, setup_rag_chain, execute_query, get_retriever

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

# 路径变量，方便之后的文件使用
file_path = './documents/LightZero_README_zh.md'

# 加载原始Markdown文档
loader = TextLoader(file_path)
orig_documents = loader.load()

# 存储对话历史
conversation_history = []

chunks = load_and_split_document(file_path, chunk_size=5000, chunk_overlap=500)
vectorstore = create_vector_store(chunks, model='OpenAI')


def rag_answer(question, temperature, k):
    """
    处理用户问题并返回答案和高亮显示的上下文

    :param question: 用户输入的问题
    :param temperature: 生成答案时使用的温度参数
    :param k: 检索到的文档块数量
    :return: 模型生成的答案和高亮显示上下文的Markdown文本
    """
    try:
        retriever = get_retriever(vectorstore, k)
        rag_chain = setup_rag_chain(model_name='kimi', temperature=temperature)

        # 将问题添加到对话历史中
        conversation_history.append(("User", question))

        # 将对话历史转换为字符串
        history_str = "\n".join([f"{role}: {text}" for role, text in conversation_history])

        retrieved_documents, answer = execute_query(retriever, rag_chain, history_str, model_name='kimi',
                                                    temperature=temperature)

        # 在文档中高亮显示上下文
        context = [retrieved_documents[i].page_content for i in range(len(retrieved_documents))]
        highlighted_document = orig_documents[0].page_content
        for i in range(len(context)):
            highlighted_document = highlighted_document.replace(context[i], f"<mark>{context[i]}</mark>")

        # 将回答添加到对话历史中
        conversation_history.append(("Assistant", answer))

        # 将对话历史存储到数据库中（此处省略数据库操作代码）

        # 返回完整的对话历史
        full_history = "\n".join([f"{role}: {text}" for role, text in conversation_history])

    except Exception as e:
        print(f"An error occurred: {e}")
        return "处理您的问题时出现错误，请稍后再试。", "", ""

    return answer, highlighted_document, full_history


def clear_context():
    """
    清除对话历史
    """
    global conversation_history
    conversation_history = []
    return "", "", ""


def export_history():
    """
    导出对话历史记录
    """
    # 从数据库中获取完整的对话历史记录（此处省略数据库操作代码）
    exported_history = "对话历史记录：\n" + "\n".join([f"{role}: {text}" for role, text in conversation_history])
    return exported_history


if __name__ == "__main__":
    with gr.Blocks(title=title, theme='ParityError/Interstellar') as zero_pal:
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column():
                inputs = gr.Textbox(
                    placeholder="请您在这里输入任何关于 LightZero 的问题。",
                    label="问题 (Q)")
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.01, step=0.01, label="温度参数")
                k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="检索到的文档块数量")
                with gr.Row():
                    gr_submit = gr.Button('提交')
                    gr_clear = gr.Button('清除上下文')

            outputs_answer = gr.Textbox(placeholder="当你点击提交按钮后，这里会显示 RAG 模型给出的回答。",
                                        label="回答 (A)")
        outputs_history = gr.Textbox(label="对话历史")
        with gr.Row():
            outputs_context = gr.Markdown(label="参考的文档，检索得到的 context 用高亮显示 (C)")
        gr_clear.click(clear_context, outputs=[outputs_context, outputs_history])
        gr_submit.click(
            rag_answer,
            inputs=[inputs, temperature, k],
            outputs=[outputs_answer, outputs_context, outputs_history],
        )
        gr.Markdown(tos_markdown)

    concurrency = int(os.environ.get('CONCURRENCY', os.cpu_count()))
    favicon_path = os.path.join(os.path.dirname(__file__), 'assets', 'avatar.png')
    zero_pal.queue().launch(max_threads=concurrency, favicon_path=favicon_path, share=True)
