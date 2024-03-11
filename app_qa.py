import os

import gradio as gr
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader

from rag_demo import load_and_split_document, create_vector_store, setup_rag_chain, execute_query

# 环境设置
load_dotenv()  # 加载环境变量
QUESTION_LANG = os.getenv("QUESTION_LANG")  # 从环境变量获取 QUESTION_LANG
assert QUESTION_LANG in ['cn', 'en'], QUESTION_LANG

if QUESTION_LANG == "cn":
    title = "LightZero RAG Demo"
    title_markdown = """
    <div align="center">
        <img src="https://raw.githubusercontent.com/puyuan1996/RAG/main/assets/banner.svg" width="80%" height="20%" alt="Banner Image">
    </div>
    <h2 style="text-align: center; color: black;"><a href="https://github.com/puyuan1996/RAG"> LightZero RAG Demo</a></h2>
    <h4 align="center"> 📢说明：请您在下面的"问题（Q）"框中输入任何关于 LightZero 的问题，然后点击"提交"按钮。右侧"回答（A）"框中会显示 RAG 模型给出的回答。在 QA 栏的下方会给出参考文档（其中检索得到的相关文段会用黄色高亮显示）。</h4>
    <h4 align="center"> 如果你喜欢这个项目，请给我们在 GitHub 点个 star ✨ 。我们将会持续保持更新。  </h4>
    <strong><h5 align="center">注意：算法模型的输出可能包含一定的随机性。相关结果不代表任何开发者和相关 AI 服务的态度和意见。本项目开发者不对生成结果作任何保证，仅供参考。<h5></strong>
    """
    tos_markdown = """
    ### 使用条款
    玩家使用本服务须同意以下条款：
    该服务是一项探索性研究预览版，仅供非商业用途。它仅提供有限的安全措施，并可能生成令人反感的内容。不得将其用于任何非法、有害、暴力、种族主义等目的。
    如果您的游玩体验有不佳之处，请发送邮件至 opendilab@pjlab.org.cn ！ 我们将删除相关信息，并不断改进这个项目。
    为了获得最佳体验，请使用台式电脑，因为移动设备可能会影响可视化效果。
    **版权所有 2024 OpenDILab。**
    """

# 路径变量，方便之后的文件使用
file_path = './documents/LightZero_README.zh.md'

# 加载原始Markdown文档
loader = TextLoader(file_path)
orig_documents = loader.load()

def rag_answer(question, model_name, temperature, embedding_model, k):
    """
    处理用户问题并返回答案和高亮显示的上下文

    :param question: 用户输入的问题
    :param model_name: 使用的语言模型名称
    :param temperature: 生成答案时使用的温度参数
    :param embedding_model: 使用的嵌入模型
    :param k: 检索到的文档块数量
    :return: 模型生成的答案和高亮显示上下文的Markdown文本
    """
    try:
        chunks = load_and_split_document(file_path)
        retriever = create_vector_store(chunks, model=embedding_model, k=k)
        rag_chain = setup_rag_chain(model_name=model_name, temperature=temperature)

        retrieved_documents, answer = execute_query(retriever, rag_chain, question, model_name=model_name, temperature=temperature)
        # 在文档中高亮显示上下文
        context = [retrieved_documents[i].page_content for i in range(len(retrieved_documents))]
        highlighted_document = orig_documents[0].page_content
        for i in range(len(context)):
            highlighted_document = highlighted_document.replace(context[i], f"<mark>{context[i]}</mark>")
    except Exception as e:
        print(f"An error occurred: {e}")
        return "处理您的问题时出现错误，请稍后再试。", ""
    return answer, highlighted_document


if __name__ == "__main__":
    with gr.Blocks(title=title, theme='ParityError/Interstellar') as rag_demo:
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column():
                inputs = gr.Textbox(
                    placeholder="请您输入任何关于 LightZero 的问题。",
                    label="问题 (Q)")
                model_name = gr.Dropdown(
                    choices=['abab6-chat', 'glm-4', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
                    value='abab6-chat',
                    label="选择语言模型")
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.01, step=0.01, label="温度参数")
                embedding_model = gr.Dropdown(
                    choices=['HuggingFace', 'TensorflowHub', 'OpenAI'],
                    value='HuggingFace',
                    label="选择嵌入模型")
                k = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="检索到的文档块数量")
                gr_submit = gr.Button('提交')

            outputs_answer = gr.Textbox(placeholder="当你点击提交按钮后，这里会显示 RAG 模型给出的回答。",
                                        label="回答 (A)")
        with gr.Row():
            outputs_context = gr.Markdown(label="参考的文档，检索得到的 context 用高亮显示 (C)")

        gr.Markdown(tos_markdown)

        gr_submit.click(
            rag_answer,
            inputs=[inputs, model_name, temperature, embedding_model, k],
            outputs=[outputs_answer, outputs_context],
        )

    concurrency = int(os.environ.get('CONCURRENCY', os.cpu_count()))
    favicon_path = os.path.join(os.path.dirname(__file__), 'assets', 'avatar.png')
    rag_demo.queue().launch(max_threads=concurrency, favicon_path=favicon_path, share=True)