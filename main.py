from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and resources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answers"
    )


# llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,
)

tools = [
    TavilySearch(
        max_results=3,
        topic="general",
        include_raw_content=False,
    )
]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


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
