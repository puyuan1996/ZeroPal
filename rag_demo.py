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
def setup_rag_chain(retriever, model_name="gpt-3.5-turbo", temperature=0):
    """设置检索增强生成流程"""
    prompt_template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Question: {question} 
    Context: {context} 
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    # 创建 RAG 链
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain


# 执行查询并打印结果
def execute_query(rag_chain, query):
    """执行查询并返回结果"""
    return rag_chain.invoke(query)


# 执行无 RAG 链的查询
def execute_query_no_rag(model_name="gpt-3.5-turbo", temperature=0, query=""):
    """执行无 RAG 链的查询"""
    llm = ChatOpenAI(model_name=model_name, temperature=temperature)
    response = llm.invoke(query)
    return response.content


# 主程序
if __name__ == "__main__":
    # 加载和分割文档
    # 下载并保存文档到本地（这里被注释掉了，因为已经假设文档存在于本地）
    # url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
    # res = requests.get(url)
    # with open("state_of_the_union.txt", "w") as f:
    #     f.write(res.text)

    # 假设文档已存在于本地
    # file_path = './documents/state_of_the_union.txt'
    file_path = './documents/LightZero_README.zh.md'

    chunks = load_and_split_document(file_path)

    # 创建向量存储
    retriever = create_vector_store(chunks)

    # 设置 RAG 流程
    rag_chain = setup_rag_chain(retriever)

    # 提出问题并获取答案
    # query = "请你分别用中英文简介 LightZero"
    # query = "请你用英文简介 LightZero"
    # query = "请你用中文简介 LightZero"
    # query = "请问 LightZero 支持哪些环境和算法，应该如何快速上手使用？请你仔细阅读我给出的 context 来回答这个问题。"
    # query = "请问 LightZero 支持 MuZero 算法在 Atari 环境上面的实现吗？请你仔细阅读我给出的 context 来回答这个问题。"
    query = "请问 LightZero 支持 AlphaZero 算法在 Atari 环境上面的实现吗？请你仔细阅读我给出的 context 来回答这个问题。"

    # 使用 RAG 链获取答案
    result_with_rag = execute_query(rag_chain, query)

    # 不使用 RAG 链获取答案
    result_without_rag = execute_query_no_rag(query=query)

    # 打印并对比两种方法的结果
    # 使用textwrap.fill来自动分段文本，width参数可以根据你的屏幕宽度进行调整
    wrapped_result_with_rag = textwrap.fill(result_with_rag, width=80)
    wrapped_result_without_rag = textwrap.fill(result_without_rag, width=80)

    # 打印自动分段后的文本
    print(f"我的问题是:\n{query}")
    print(f"Result with RAG:\n{wrapped_result_with_rag}")
    print(f"Result without RAG:\n{wrapped_result_without_rag}")