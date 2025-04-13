# simple_rag.py  
# 一个简单的基于LangChain的RAG问答系统  

import os  
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.embeddings import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS  
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline  
from langchain.prompts import PromptTemplate  
from langchain.chains import RetrievalQA  
from transformers import pipeline  
import torch

# 步骤1: 加载PDF文件  
pdf_path = "Lecture\L1_Introduction.pdf"  # 确保此文件在当前目录下  
loader = PyPDFLoader(pdf_path)  
documents = loader.load()  
print(f"加载了 {len(documents)} 页PDF。")  

# 步骤2: 切分文档  
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  
chunks = text_splitter.split_documents(documents)  
print(f"文档被分割成 {len(chunks)} 个块。")  

# 步骤3: 初始化嵌入模型  
embeddings = HuggingFaceEmbeddings(  
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},  
)  

# 步骤4: 创建向量存储  
# 检查是否存在已保存的向量存储  
vector_store_path = "vector_store"  
if os.path.exists(vector_store_path):  
    print("加载现有向量存储...")  
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)  
else:  
    print("创建新的向量存储...")  
    vector_store = FAISS.from_documents(chunks, embeddings)  
    # 保存向量存储以便将来使用  
    vector_store.save_local(vector_store_path)  

# 步骤5: 初始化语言模型  
# 使用小型模型，适合本地运行  
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-large",  # 更大的T5模型
    task="text2text-generation",
    model_kwargs={"temperature": 0.7},  # 使用GPU加速
    pipeline_kwargs={"max_new_tokens": 500}
)
# 步骤6: 创建检索QA链
template = """
你是一个问答机器人，你的任务是根据提供的上下文回答用户的问题。
请仔细阅读上下文，并从中提取出与问题相关的答案。
如果上下文没有提供答案，请回答“我不知道”。
请不要编造答案。

上下文: {context}

问题: {question}

答案:
"""
PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # 增加检索的文档块数量 
qa_chain = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  # 使用stuff方法将所有上下文一次性传给LLM  
    retriever=retriever,  
    return_source_documents=True,  
    chain_type_kwargs={"prompt": PROMPT}  
)  

# 步骤7: 问答交互  
def ask_question(question):  
    result = qa_chain({"query": question})  
    answer = result["result"]  
    sources = result["source_documents"]  
    
    print("\n回答:")  
    print(answer)  
    print("\n参考来源:")  
    for i, source in enumerate(sources):  
        print(f"来源 {i+1}:")  
        print(f"- 页码: {source.metadata.get('page', 'N/A')}")  
        print(f"- 内容片段: {source.page_content[:150]}...")  
        print()  

# 开始问答交互  
if __name__ == "__main__":  
    print("\n简单RAG问答系统已准备就绪!")  
    print("输入问题或输入'退出'结束程序")  
    
    while True:  
        question = input("\n请输入问题: ")  
        if question.lower() in ['退出', 'exit', 'quit']:  
            break  
        if question:  
            ask_question(question)  