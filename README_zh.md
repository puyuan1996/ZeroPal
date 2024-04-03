# RAG Demo 使用说明

简体中文 | [English](https://github.com/puyuan1996/RAG/blob/main/README.md) 

## 简介

RAG 是一个基于检索增强生成 (RAG) 的问答系统示例项目。
- 它使用大型语言模型（如 GPT-3.5）和文档检索向量数据库（如 Weaviate）来响应用户的问题，通过检索相关的文档上下文以及利用语言模型的生成能力来提供准确的回答。
- 同时提供了一个基于 Gradio 和 rag_demo.py 构建的网页交互式应用。

## rag_demo.py 功能

- 支持通过环境变量加载 OpenAI API 密钥。
- 支持加载本地文档并将其分割成小块。
- 支持创建向量存储，并将文档块转换为向量存储在 Weaviate 中。
- 支持设置检索增强生成流程，结合文档检索和语言模型生成对用户问题进行回答。
- 支持执行查询并打印结果，可以选择是否通过 RAG 流程。

## app.py 功能

- 创建一个Gradio应用，用户可以在其中输入问题，应用会使用Retrieval-Augmented Generation (RAG)模型来寻找答案并将结果显示在界面上。
- 其中，检索到的上下文会在Markdown文档中高亮显示，帮助用户理解答案的来源。应用界面分为两部分：顶部是问答区，底部展示了RAG模型参考的上下文。

## 使用方法

1. 克隆项目到本地。
2. 安装依赖。

```shell
pip3 install -r requirements.txt
```
3. 在项目根目录下创建 `.env` 文件，并添加你的 OpenAI API 密钥：

```
OPENAI_API_KEY='你的API密钥'
QUESTION_LANG='cn' # 问题语言，目前可选值为 'cn'
```

4. 确保已经有可用的文档作为上下文，或者使用注释掉的代码段下载你需要参考的文档。
5. 执行 `python3 -u rag_demo.py` 文件即可开始使用。

## 示例

```python

if __name__ == "__main__":
    # 假设文档已存在于本地
    file_path = './documents/LightZero_README_zh.md'
    # 加载和分割文档
    chunks = load_and_split_document(file_path)
    # 创建向量存储
    retriever = create_vector_store(chunks)
    # 设置 RAG 流程
    rag_chain = setup_rag_chain()
    
    # 提出问题并获取答案
    query = "请问 LightZero 里面实现的 AlphaZero 算法支持在 Atari 环境上运行吗？请详细解释原因"
    # 使用 RAG 链获取参考的文档与答案
    retrieved_documents, result_with_rag = execute_query(retriever, rag_chain, query)
    # 不使用 RAG 链获取答案
    result_without_rag = execute_query_no_rag(query=query)
    
    # 此处省略部分数据处理代码，具体细节请参考本仓库中的源文件
    
    # 打印并对比两种方法的结果
    print("=" * 40)
    print(f"我的问题是:\n{query}")
    print("=" * 40)
    print(f"Result with RAG:\n{wrapped_result_with_rag}\n检索得到的context是: \n{context}")
    print("=" * 40)
    print(f"Result without RAG:\n{wrapped_result_without_rag}")
    print("=" * 40)
```

## 项目结构

```
RAG/
│
├── rag_demo.py            # RAG 演示脚本，支持输出检索到的文档块。
├── app_qa.py              # 基于 Gradio 和 rag_demo.py 构建的网页交互式应用。
├── app_mqa.py             # 基于 Gradio 和 rag_demo.py 构建的网页交互式应用。支持保持对话历史。
├── .env                   # 环境变量配置文件
└── documents/             # 文档文件夹
    └── your_document.txt  # 上下文文档
```

## 贡献指南

如果您希望为 RAG 贡献代码，请遵循以下步骤：

1. Fork 项目。
2. 创建一个新的分支。
3. 提交你的改动。
4. 提交 Pull Request。

## 问题和支持

如果遇到任何问题或需要帮助，请通过项目的 Issues 页面提交问题。

## 许可证

本仓库中的所有代码都符合 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)。
