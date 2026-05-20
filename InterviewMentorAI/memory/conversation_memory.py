from langchain_classic.memory import ConversationBufferMemory


def create_conversation_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
