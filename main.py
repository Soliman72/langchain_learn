from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
tools = [
    TavilySearch(
        max_results=3,
        topic="general",
        include_raw_content=False,
    )
]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Find 3 AI Engineer jobs using LangChain in the Bay Area.Return only:- Company- Job title- Location- URL"
                )
            ]
        }
    )
    print(result)


if __name__ == "__main__":
    main()
