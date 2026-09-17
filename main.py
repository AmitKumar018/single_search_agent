import os

import requests
import streamlit as st
import certifi
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from langchain import hub

from langchain_community.tools.tavily_search import TavilySearchResults


os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

search_tool = TavilySearchResults(max_results=2)


    
    
@tool
def get_weather_data(city: str) -> str:
    """Fetch the current temperature in that city"""

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)
    if response.status_code != 200:
        return "Information not available"

    data = response.json()  # <-- fixed: call it

    if "current" not in data:
        return "Information not available"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)



prompt = hub.pull("hwchase17/react")

tools=[search_tool,get_weather_data]

agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  
)



user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather"
)


# stramlit

if st.button("Run Agent"):
    if user_query:
        with st.spinner("Agent is thinking..."):
            try:
                response = agent_executor.invoke({
                    "input": user_query
                })

                st.success("Response Generated")
                st.markdown("## Final Response")
                st.write(response["output"])
            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")