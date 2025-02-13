# LOAD

import langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from gnews import GNews
import requests
import os

os.environ["OPENAI_API_KEY"] = "MY KEY"

os.environ["ALPHA_VANTAGE_API_KEY"] = " MY KEY"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

# Task functions

def get_stock_data(symbol):
    url = f" {symbol} {ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()


def get_financial_news():
    google_news = GNews(language="en", country= "US", max_results=2)
    try:
        news_articles = google_news.get_news("stock market")
        if not news_articles:
            return{"error: No news found"}
    except Exception as e:
        return {"error: API"}
    
# Tools

stock_tool = Tool(
            name = "Stock Data Fetcher",
            func = lambda symbol: get_stock_data(symbol),
            description = "Fetches real-time stock data for a given stock symbol"
            )

news_tool = Tool(
            name = "Financial News Fetcher",
            func = lambda _: get_financial_news(),
            description = "Retrieves the latest financial news headlines"
            )



# Agent

llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=500)
agent= initialize_agent(
            tools= [stock_tool, news_tool],
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
            )

# Analyze


def analyze_market():
    chat_history = []
    while True:
        user_query = input("Ask the financial analysis agent:")
        reponse = agent.invoke("input": user_query, "chat_history":chat_history)
        chat_history.append(user_query, response)
        print(f"Response: {response}")


if __name__ == "__main__":
    analyze_market()
