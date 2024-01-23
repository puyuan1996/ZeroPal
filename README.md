# RAG Demo

English | [简体中文(Simplified Chinese)](https://github.com/puyuan1996/RAG/blob/main/README_zh.md) 

## Introduction

RAG is a demonstration project for a question-answering system based on Retrieval-Augmented Generation (RAG). 
- It utilizes large language models such as GPT-3.5 in conjunction with a document retrieval vector database like Weaviate to respond to user queries by retrieving relevant document contexts and leveraging the generative capabilities of the language model.
- The project also includes a web-based interactive application built with Gradio and rag_demo.py.

## rag_demo.py Features

- Supports loading OpenAI API keys via environment variables.
- Facilitates loading local documents and splitting them into chunks.
- Allows for the creation of a vector store and the conversion of document chunks into vectors for storage in Weaviate.
- Sets up a Retrieval-Augmented Generation process, combining document retrieval and language model generation to answer user questions.
- Executes queries and prints results, with the option to use the RAG process or not.

## app.py Features

- Creates a Gradio application where users can input questions and the application employs the Retrieval-Augmented Generation (RAG) model to find answers, displaying results within the interface.
- Retrieved contexts are highlighted in the Markdown document to help users understand the source of the answers. The application interface is divided into two sections: the top for Q&A and the bottom to display the contexts referred to by the RAG model.

## How to Use

1. Clone the project to your local machine.
2. Install dependencies.

```shell
pip3 install -r requirements.txt
```
3. Create a `.env` file in the project root directory and add your OpenAI API key:

```
OPENAI_API_KEY='your API key'
QUESTION_LANG='cn' # The language of the question, currently available option is 'cn'
```

4. Ensure you have available documents as context or use the commented-out code snippet to download the documents you want to reference.
5. Run the `python3 -u rag_demo.py` file to start using the application.

## Example

```python

# The difference between rag_demo.py and rag_demo_v0.py is that it can output the retrieved document chunks.
if __name__ == "__main__":
    # Assuming documents are already present locally
    file_path = './documents/LightZero_README.zh.md'
    # Load and split document
    chunks = load_and_split_document(file_path)
    # Create vector store
    retriever = create_vector_store(chunks)
    # Set up RAG process
    rag_chain = setup_rag_chain()
    
    # Pose a question and get an answer
    query = "Does the AlphaZero algorithm implemented in LightZero support running in the Atari environment? Please explain in detail."
    # Use RAG chain to get referenced documents and answer
    retrieved_documents, result_with_rag = execute_query(retriever, rag_chain, query)
    # Get an answer without using RAG chain
    result_without_rag = execute_query_no_rag(query=query)
    
    # Details of data handling code are omitted here, please refer to the source files in this repository for specifics
    
    # Print and compare results from both methods
    print("=" * 40)
    print(f"My question is:\n{query}")
    print("=" * 40)
    print(f"Result with RAG:\n{wrapped_result_with_rag}\nRetrieved context is: \n{context}")
    print("=" * 40)
    print(f"Result without RAG:\n{wrapped_result_without_rag}")
    print("=" * 40)
```

## Project Structure

```
RAG/
│
├── rag_demo_v0.py         # RAG demonstration script without support for outputting retrieved document chunks.
├── rag_demo.py            # RAG demonstration script with support for outputting retrieved document chunks.
├── app.py                 # Web-based interactive application built with Gradio and rag_demo.py.
├── .env                   # Environment variable configuration file
└── documents/             # Documents folder
    └── your_document.txt  # Context document
```

## Contribution Guide

If you would like to contribute code to RAG, please follow these steps:

1. Fork the project.
2. Create a new branch.
3. Commit your changes.
4. Submit a Pull Request.

## Issues and Support

If you encounter any issues or require assistance, please submit a problem through the project's Issues page.

## License

All code in this repository is compliant with [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).