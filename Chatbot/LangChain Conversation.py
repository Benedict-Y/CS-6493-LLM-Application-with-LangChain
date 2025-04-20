from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

def run_langchain_conversation():
    chat_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_base="https://api.chatanywhere.tech/v1",
        openai_api_key="YOUR API KEY"
        #  Free for deepseek-r1，deepseek-v3，gpt-3.5-turbo，gpt-4o-mini，gpt-4o
    )
    chain = ConversationChain(llm=chat_llm, memory=ConversationBufferMemory())
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        response = chain.run(user_input)
        print("Assistant:", response)

if __name__ == '__main__':
    run_langchain_conversation()
