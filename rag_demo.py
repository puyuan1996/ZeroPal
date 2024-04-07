"""
参考博客：https://mp.weixin.qq.com/s/RUdZjQMSlVOfHfhErSNXnA
"""
# 导入必要的库与模块
import json
import os
import textwrap

import requests
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, TensorflowHubEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from weaviate import Client
from weaviate.embedded import EmbeddedOptions
from zhipuai import ZhipuAI
from openai import AzureOpenAI

# 环境设置与文档下载
load_dotenv()  # 加载环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量获取 OpenAI API 密钥
MIMIMAX_API_KEY = os.getenv("MIMIMAX_API_KEY")
MIMIMAX_GROUP_ID = os.getenv("MIMIMAX_GROUP_ID")
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
KIMI_OPENAI_API_KEY = os.getenv("KIMI_OPENAI_API_KEY")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

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

    if model == "OpenAI":
        embedding_model = OpenAIEmbeddings()
    elif model == "HuggingFace":
        embedding_model = HuggingFaceEmbeddings()
    elif model == "TensorflowHub":
        embedding_model = TensorflowHubEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding model: {model}")

    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embedding_model,
        by_text=False
    )
    return vectorstore


def get_retriever(vectorstore, k=4):
    return vectorstore.as_retriever(search_kwargs={'k': k})


def setup_rag_chain(model_name="gpt-4", temperature=0):
    """设置检索增强生成流程"""
    if model_name.startswith("gpt"):
        # 如果是以gpt开头的模型,使用原来的逻辑
        prompt_template = """您是一个用于问答任务的专业助手。
        在处理问答任务时,请根据所提供的[上下文信息]给出回答。
        如果[上下文信息]与[问题]不相关,那么请运用您的知识库为提问者提供准确的答复。
        请确保回答内容的质量, 包括相关性、准确性和可读性。
        [问题]: {question} 
        [上下文信息]: {context} 
        [回答]:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        # 创建 RAG 链,参考 https://python.langchain.com/docs/expression_language/
        rag_chain = (
                prompt
                | llm
                | StrOutputParser()
        )
    else:
        # 如果不是以gpt开头的模型,返回None
        rag_chain = None
    return rag_chain


# 执行查询并打印结果
def execute_query(retriever, rag_chain, query, model_name="gpt-4", temperature=0):
    """
    执行查询并返回结果及检索到的文档块

    参数:
    retriever: 文档检索器对象
    rag_chain: 检索增强生成链对象,如果为None则不使用RAG链
    query: 查询问题
    model_name: 使用的语言模型名称,默认为"gpt-4"
    temperature: 生成温度,默认为0

    返回:
    retrieved_documents: 检索到的文档块列表
    response_text: 生成的回答文本
    """
    # 使用检索器检索相关文档块
    retrieved_documents = retriever.invoke(query)

    if rag_chain is not None:
        # 如果有RAG链,则使用RAG链生成回答
        rag_chain_response = rag_chain.invoke({"context": retrieved_documents, "question": query})
        response_text = rag_chain_response
    else:
        # 如果没有RAG链,则将检索到的文档块和查询问题按照指定格式输入给语言模型
        if model_name == "kimi":
            # 对于有检索能力的模型,使用不同的模板
            prompt_template = """您是一个用于问答任务的专业助手。
            在处理问答任务时，请根据所提供的【上下文信息】和【你的知识库和检索到的相关文档】给出回答。
            请确保回答内容的质量，包括相关性、准确性和可读性。
           【问题】: {question} 
           【上下文信息】: {context} 
           【回答】:
            """
        else:
            prompt_template = """您是一个用于问答任务的专业助手。
            在处理问答任务时，请根据所提供的【上下文信息】给出回答。
            如果【上下文信息】与【问题】不相关,那么请运用您的知识库为提问者提供准确的答复。
            请确保回答内容的质量，包括相关性、准确性和可读性。
           【问题】: {question} 
           【上下文信息】: {context} 
           【回答】:
            """

        context = '\n'.join(
            [f'**Document {i}**: ' + retrieved_documents[i].page_content for i in range(len(retrieved_documents))])
        prompt = prompt_template.format(question=query, context=context)
        response_text = execute_query_no_rag(model_name=model_name, temperature=temperature, query=prompt)
    return retrieved_documents, response_text


def execute_query_no_rag(model_name="gpt-4", temperature=0, query=""):
    """执行无 RAG 链的查询"""
    if model_name.startswith("gpt"):
        # 如果是以gpt开头的模型,使用原来的逻辑
        llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        response = llm.invoke(query)
        return response.content
    elif model_name.startswith("azure_gpt"):
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version="2024-02-15-preview"
        )
        message_text = [{"role": "user", "content": query}, ]
        completion = client.chat.completions.create(
            model=model_name[6:],  # model_name = 'azure_gpt-4', 'azure_gpt-35-turbo-16k', 'azure_gpt-35-turbo'
            messages=message_text,
            temperature=temperature,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        return completion.choices[0].message.content
    elif model_name == 'abab6-chat':
        # 如果是'abab6-chat'模型,使用专门的API调用方式
        url = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=" + MIMIMAX_GROUP_ID
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + MIMIMAX_API_KEY}
        payload = {
            "bot_setting": [
                {
                    "bot_name": "MM智能助理",
                    "content": "MM智能助理是一款由MiniMax自研的,没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司,一直致力于进行大模型相关的研究。",
                }
            ],
            "messages": [{"sender_type": "USER", "sender_name": "小明", "text": query}],
            "reply_constraints": {"sender_type": "BOT", "sender_name": "MM智能助理"},
            "model": model_name,
            "tokens_to_generate": 1034,
            "temperature": temperature,
            "top_p": 0.9,
        }

        response = requests.request("POST", url, headers=headers, json=payload)
        # 将 JSON 字符串解析为字典
        response_dict = json.loads(response.text)
        # 提取 'reply' 键对应的值
        return response_dict['reply']

    elif model_name == 'glm-4':
        # 如果是'glm-4'模型,使用专门的API调用方式
        client = ZhipuAI(api_key=ZHIPUAI_API_KEY)  # 填写您自己的APIKey
        response = client.chat.completions.create(
            model=model_name,  # 填写需要调用的模型名称
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content
    elif model_name == 'kimi':
        # 如果是'kimi'模型,使用专门的API调用方式
        from openai import OpenAI
        client = OpenAI(
            api_key=KIMI_OPENAI_API_KEY,
            base_url="https://api.moonshot.cn/v1",
        )
        messages = [
            {
                "role": "system",
                "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。",
            },
            {"role": "user",
             "content": query},
        ]
        completion = client.chat.completions.create(
            model="moonshot-v1-128k",
            messages=messages,
            temperature=0.01,
            top_p=1.0,
            n=1,  # 为每条输入消息生成多少个结果
            stream=False  # 流式输出
        )
        return completion.choices[0].message.content
    else:
        # 如果模型不支持,抛出异常
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    # 假设文档已存在于本地
    file_path = './documents/LightZero_README_zh.md'
    # model_name = "glm-4"  # model_name=['abab6-chat', 'glm-4', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'azure_gpt-4', 'azure_gpt-35-turbo-16k', 'azure_gpt-35-turbo']
    # model_name = 'azure_gpt-4'
    model_name = 'kimi'
    temperature = 0.01
    embedding_model = 'OpenAI'  # embedding_model=['HuggingFace', 'TensorflowHub', 'OpenAI']

    # 加载和分割文档
    chunks = load_and_split_document(file_path, chunk_size=5000, chunk_overlap=500)

    # 创建向量存储
    vectorstore = create_vector_store(chunks, model=embedding_model)
    retriever = get_retriever(vectorstore, k=4)

    # 设置 RAG 流程
    rag_chain = setup_rag_chain(model_name=model_name, temperature=temperature)

    # 提出问题并获取答案
    query = ("请回答下面的问题：（1）请简要介绍一下 LightZero。（2）请详细介绍 LightZero 的框架结构。 （3）请给出安装 LightZero，运行他们的示例代码的详细步骤。（4）- 请问 LightZero 具体支持什么任务（tasks/environments）? （5）请问 LightZero 具体支持什么算法?（6）请问 LightZero 具体支持什么算法，各自支持在哪些任务上运行? （7）请问 LightZero 里面实现的 MuZero 算法支持在 Atari 任务上运行吗？（8）请问 LightZero 里面实现的 AlphaZero 算法支持在 Atari 任务上运行吗？（9）LightZero 支持哪些算法? 各自的优缺点是什么? 我应该如何根据任务特点进行选择呢？（10）请结合 LightZero 中的代码介绍他们是如何实现 MCTS 的。（11）请问对这个仓库提出详细的改进建议")
    """
    （1）请简要介绍一下 LightZero。 
    （2）请详细介绍 LightZero 的框架结构。 
    （3）请给出安装 LightZero，运行他们的示例代码的详细步骤 。
    （4）请问 LightZero 具体支持什么任务（tasks/environments）? 
    （5）请问 LightZero 具体支持什么算法?
    （6）请问 LightZero 具体支持什么算法，各自支持在哪些任务上运行? 
    （7）请问 LightZero 里面实现的 MuZero 算法支持在 Atari 任务上运行吗？
    （8）请问 LightZero 里面实现的 AlphaZero 算法支持在 Atari 任务上运行吗？
    （9）LightZero 支持哪些算法? 各自的优缺点是什么? 我应该如何根据任务特点进行选择呢？
    （10）请结合 LightZero 中的代码介绍他们是如何实现 MCTS 的。
    （11）请问对这个仓库提出详细的改进建议。
    """

    # 使用 RAG 链获取参考的文档与答案
    retrieved_documents, result_with_rag = execute_query(retriever, rag_chain, query, model_name=model_name,
                                                         temperature=temperature)

    # 不使用 RAG 链获取答案
    result_without_rag = execute_query_no_rag(model_name=model_name, query=query, temperature=temperature)

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
