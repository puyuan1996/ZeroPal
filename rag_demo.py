"""
参考博客：https://mp.weixin.qq.com/s/RUdZjQMSlVOfHfhErSNXnA
"""
# 导入必要的库与模块
import os
import textwrap

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from weaviate import Client
from weaviate.embedded import EmbeddedOptions

# 环境设置与文档下载
load_dotenv()  # 加载环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量获取 OpenAI API 密钥

# 确保 OPENAI_API_KEY 被正确设置
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found in the environment variables.")


# 文档加载与分割
def load_and_split_document(file_path, chunk_size=500, chunk_overlap=50):
    """加载文档并分割成小块"""
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks


# 向量存储建立
def create_vector_store(chunks, model="OpenAI"):
    """将文档块转换为向量并存储到 Weaviate 中"""
    client = Client(embedded_options=EmbeddedOptions())
    embedding_model = OpenAIEmbeddings() if model == "OpenAI" else None  # 可以根据需要替换为其他嵌入模型
    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embedding_model,
        by_text=False
    )
    return vectorstore.as_retriever()


# 定义检索增强生成流程
def setup_rag_chain(model_name="gpt-4", temperature=0):
    """设置检索增强生成流程"""
    # prompt_template = """You are a professional assistant for question-answering tasks.
    # When handling question-answering tasks, please provide relevant answers based on the provided context information.
    # If the context information is not relevant to the question, please use your knowledge base to provide accurate replies to the inquirer.
    # Please ensure the quality of the answers, including accuracy, relevance, readability, and comprehensibility.
    # Question: {question}
    # Context: {context}
    # Answer:
    # """
    prompt_template = """你是一个用于问答任务的专业助手。
    在处理问答任务时，请根据所提供的上下文信息给出相关答案。
    如果上下文信息与问题不相关，那么请运用您的知识库为提问者提供准确的答复。
    请确保回答内容的质量，包括准确性、相关性、可读性和易理解性。
    问题: {question} 
    上下文: {context} 
    回答:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    # 创建 RAG 链，参考 https://python.langchain.com/docs/expression_language/
    rag_chain = (
            prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


# 执行查询并打印结果
def execute_query(retriever, rag_chain, query):
    """执行查询并返回结果及检索到的文档块"""
    retrieved_documents = retriever.invoke(query)
    rag_chain_response = rag_chain.invoke({"context": retrieved_documents, "question": query})
    return retrieved_documents, rag_chain_response


# 执行无 RAG 链的查询
def execute_query_no_rag(model_name="gpt-4", temperature=0, query=""):
    """执行无 RAG 链的查询"""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    response = llm.invoke(query)
    return response.content


# rag_demo.py 相对 rag_demo_v0.py 的不同之处在于可以输出检索到的文档块。
if __name__ == "__main__":
    # 假设文档已存在于本地
    file_path = './documents/LightZero_README.zh.md'

    # 加载和分割文档
    chunks = load_and_split_document(file_path)

    # 创建向量存储
    retriever = create_vector_store(chunks)

    # 设置 RAG 流程
    rag_chain = setup_rag_chain()

    # 提出问题并获取答案
    query = "请问 LightZero 里面实现的 AlphaZero 算法支持在 Atari 环境上运行吗？请详细解释原因"
    # query = "请详细解释 MCTS 算法的原理，并给出带有详细中文注释的 Python 代码示例"

    # 使用 RAG 链获取参考的文档与答案
    retrieved_documents, result_with_rag = execute_query(retriever, rag_chain, query)

    # 不使用 RAG 链获取答案
    result_without_rag = execute_query_no_rag(query=query)

    # 打印并对比两种方法的结果
    # 使用textwrap.fill来自动分段文本，width参数可以根据你的屏幕宽度进行调整
    wrapped_result_with_rag = textwrap.fill(result_with_rag, width=80)
    wrapped_result_without_rag = textwrap.fill(result_without_rag, width=80)
    context = '\n'.join(
        [f'**Document {i}**: ' + retrieved_documents[i].page_content for i in range(len(retrieved_documents))])

    # 打印自动分段后的文本
    print("=" * 40)
    print(f"我的问题是:\n{query}")
    print("=" * 40)
    print(f"Result with RAG:\n{wrapped_result_with_rag}\n检索得到的context是: \n{context}")
    print("=" * 40)
    print(f"Result without RAG:\n{wrapped_result_without_rag}")
    print("=" * 40)
