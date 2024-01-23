"""
此代码的整体功能是构建一个Gradio界面，该界面允许用户输入问题，然后使用一个Retrieval-Augmented Generation (RAG)模型来找到并显示答案。
同时，该界面通过高亮显示检索到的上下文来展示RAG模型如何找到答案。
界面被分为两个部分：左边是问答区域，右边是原始Markdown文档，其中包含了RAG模型参考的上下文。

结构概述：
- 导入Gradio库和RAG相关的函数和类。
- 加载并切分文档，设置检索器和RAG模型。
- 定义了一个rag_answer函数，用于处理用户的问题，并返回模型的答案和高亮显示的上下文。
- 使用Gradio的Interface构建用户界面，设置输入框和输出框的属性，并定义了CSS样式来改善界面外观。
- 启动Gradio界面并设置为可以分享。
"""

import gradio as gr
from rag_demo_v2 import load_and_split_document, create_vector_store, setup_rag_chain_v2, execute_query_v2
from langchain.document_loaders import TextLoader

# 路径变量，方便之后的文件使用
file_path = './documents/LightZero_README.zh.md'
chunks = load_and_split_document(file_path)
retriever = create_vector_store(chunks)
rag_chain = setup_rag_chain_v2()

# 加载原始Markdown文档
loader = TextLoader(file_path)
orig_documents = loader.load()


def rag_answer(question):
    retrieved_documents, answer = execute_query_v2(retriever, rag_chain, question)
    # Highlight the context in the document
    context = [retrieved_documents[i].page_content for i in range(len(retrieved_documents))]
    highlighted_document = orig_documents[0].page_content
    for i in range(len(context)):
        highlighted_document = highlighted_document.replace(context[i], f"<mark>{context[i]}</mark>")
    return answer, highlighted_document


rag_demo = gr.Interface(
    # 绑定上面定义的RAG回答函数
    fn=rag_answer,
    # 设置输入框，包括占位符和标签
    inputs=gr.Textbox(
        placeholder="请您输入任何关于 LightZero 的问题。📢右侧上栏会给出 RAG 模型给出的回答，右侧下栏会给出参考文档（检索得到的 context 用黄色高亮显示）。",
        label="问题 (Q)"),
    # 设置输出框，包括答案和高亮显示参考文档
    outputs=[
        gr.Markdown(label="回答 (A)"),
        gr.Markdown(label="参考的文档，检索得到的 context 用高亮显示"),
    ],
    # 自定义CSS样式
    css='''
        .output_text { background-color: #e8f4fc; } /* 设置答案框的背景色 /
        .input_text { background-color: #ffefc4; } / 设置问题框的背景色 /
        mark { background-color: yellow; } / 设置高亮文本的背景色 */
    ''',
    live=False,  # 设置为非实时模式，用户需要点击提交按钮
)

if __name__ == "__main__":
    # 启动界面，设置为可以分享。如果分享公网链接失败，可以在本地执行 ngrok http 7860 将本地端口映射到公网
    rag_demo.launch(share=True)
