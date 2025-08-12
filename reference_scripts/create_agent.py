import os
import dotenv

from pydantic import BaseModel

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt.chat_agent_executor import AgentState

dotenv.load_dotenv() 
# or
envs = dotenv.dotenv_values()

# configure stuctured output with Pydantic model or TypeDict
class WeatherResponse(BaseModel):
    conditions: str
    #model_config = ConfigDict(coerce_numbers_to_str=True)

def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

# configure an LLM with specific parameters
model = init_chat_model(
    "gemini-2.5-flash",
    temperature=0
)
# custom prompt
def prompt(state: AgentState, config: RunnableConfig) -> list[AnyMessage]:  
    user_name = config["configurable"].get("user_name")
    system_msg = f"You are a helpful assistant. Address the user as {user_name}."
    return [{"role": "system", "content": system_msg}] + state["messages"]
# Prompts instruct the LLM how to behave. Add one of the following types of prompts:
# Static: A string is interpreted as a system message.
# Dynamic: A list of messages generated at runtime, based on input or configuration.

# create an agent
agent = create_react_agent(
    #model="anthropic:claude-3-7-sonnet-latest",  
    #model="gemini-2.5-flash",
    model=model,
    tools=[get_weather],  
    #prompt="You are a helpful assistant",
    prompt=prompt,
    response_format=WeatherResponse,
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

# print(response.keys())
response["structured_response"]
# "structured_response" key:value is created upon the response_format parameter, otherwise only "messages".


# allow multi-turn conversations with an agent, you need to enable persistence by providing a checkpointer when creating an agent. At runtime, you need to provide a config containing thread_id — a unique identifier for the conversation (session)
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

agent = create_react_agent(
    model="gemini-2.5-flash",
    tools=[get_weather],
    checkpointer=checkpointer  
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}
sf_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    config  
)
ny_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)





agent = create_react_agent(
    model="gemini-2.5-flash",
    tools=[get_weather],
)

# stream mode options: 
stream_mode = ["values", "updates", "custom", "messages", "debug"]

for chunk in agent.stream(
    input={"messages": [{"role": "user", "content": "what is the weather in sf"}]},
    stream_mode=stream_mode[3] 
):
    print(chunk)
    print("\n")


