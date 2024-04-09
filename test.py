from langchain.agents import initialize_agent, AgentType, tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    return len(text)


if __name__ == "__main__":
    llm = ChatOpenAI()
    agent_executer: AgentExecutor = initialize_agent(
        tools=[get_text_length],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    agent_executer.invoke({"input": "What is the length of the text Dog?"})
