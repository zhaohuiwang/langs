
"""
Chat Model: Handles message-based input/output, supports features like tool calling or streaming, but lacks autonomous decision-making or state management. init_chat_model() function in LangChain initializes a chat model. It returns a BaseChatModel object or a configurable model emulator if no default model is specified.

Agent: Combines a chat model with tools, memory (e.g., conversation history), and logic to make decisions, execute actions, or manage workflows. Agents are often built with frameworks like LangGraph for more complex behavior. create_react_agent() function in LangChin create an agent.


The language model is designed to determine whether a tool is needed based on the user's query or the context of the conversation. This decision is typically based on the model's understanding of the query and the tool descriptions or schemas provided during binding. If the model determines that a tool is relevant, it outputs a structured response (often in JSON format) specifying which tool to call and with what arguments. LangChain then handles the execution of the tool based on this output.
Key points: Tool Schema or Metadata help to guide the LLM on when and how to call a tool.
Dynamic Decision: THe LLM denamically decide whether to call a tool based on the input and its internal reasoning. For example, if the query is "What is 2+2?" the LLM might not need to call any tool. But for "What is weather today?", it might call a search tool to fetch relevant data.




"""
# import dotenv
# evns = dotenv.dotenv_values(".env") 


import os
import dotenv
import json


from typing import Annotated
from typing_extensions import TypedDict


from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch

dotenv.load_dotenv() 



### Define the web search tool - Tavily Search Engine
tool = TavilySearch(max_results=2)
tools = [tool]
# tool.invoke("What's a 'node' in LangGraph?")

# select a language model and initiate a chat model
llm = init_chat_model("gemini-2.5-flash")

# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

# in-memory checkpointer for tutorial. In a production application, you would likely change this to use SqliteSaver or PostgresSaver and connect a database.
memory = InMemorySaver()


### Create a StateGraph
class State(TypedDict):
    # Messages have the type "list". The `add_messages` function in the annotation defines how this state key should be updated (in this case, it appends messages to the list, rather than overwriting them, default reducer)
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    #return {"messages": [llm.invoke(state["messages"])]}
    # with the llm only, the bot's knowledge is limited to what's in its training data. We can expand the bot's knowledge and make it more capable with tools.
    return {"messages": [llm_with_tools.invoke(state["messages"])]}
    #  LangChain's ChatModel accepts a list of Message objects as inputs. These messages come in a variety of forms such as `HumanMessage`` (user input) or `AIMessage`` (LLM response). You can manually update messages in your graph state (e.g. human-in-the-loop).


# StateGraph object defines the structure of our chatbot as a "state machine"
graph_builder = StateGraph(State)

### Add Node
# Nodes and Edges are nothing more than functions. In short: nodes do the work, edges tell what to do next.
# nodes represent the llm and functions our chatbot can call and edges to specify how the bot should transition between these functions.
graph_builder.add_node("chatbot", chatbot)
# The first argument is the unique node name. The second argument is function (callable) or runnable that will be called whenever the node is used.

# the chatbot node function takes the current State as input and returns a dictionary containing an updated messages list under the key "messages". This is the basic pattern for all LangGraph node functions.The add_messages function in our State will append the LLM's response messages to whatever messages are already in the state.

# Node Caching: `cache_policy=CachePolicy(ttl=3))` 
# from langgraph.cache.memory import InMemoryCache
# graph_builder.add_node(name, value, cache_policy=CachePolicy(ttl=3))
# ttl is time to live in second by default. This is useful for optimizing performance by avoiding redundant computations for expensive operations while ensuring data doesn't stay stale for too long.


### <replaceable-1>
### Create a function to run the tools
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
tool_node = BasicToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)
### </replaceable-1>

### <replaceable-1>
# you can use LangGraph's prebuilt ToolNode if you donot want build yourself
# tool_node = ToolNode(tools=[tool])
# graph_builder.add_node("tools", tool_node)
### </replaceable-1>

### <replaceable-2>
### Define the conditional_edges
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# LangChain always deserize the state updates into the Message when using `add_messages`. You can use the dot notation to access message attribute, like  state["messages"][-1].<key> [-1] specifies the the most recent 

    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node. It defaults to the identity function, but if you want to use a node named something else apart from "tools", uou can update the value of the dictionary to something else e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
### </replaceable-2>

### <replaceable-2>
# You can replace this with the prebuilt tools_condition to be more concise.
# graph_builder.add_conditional_edges(
#     "chatbot",
#     tools_condition,
#)
### </replaceable-2>


# graph.add_edge(START, "node_a") # Entry point langgraph.graph.START
# graph.add_conditional_edges(START, routing_function) # conditional entry point
# graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})
# graph.add_edge("node_a", "node_b") # If you always want to go from node A to node B
# graph.add_conditional_edges("node_a", routing_function) # Conditional Edges

# map-reduce design patterns (Send class) where your graph invokes the same node multiple times in parallel with different states, before aggregating the results back into the main graph's state.

# Command class to combine control flow (edges) and state updates (nodes)
# def my_node(state: State) -> Command[Literal["my_other_node"]]:
#     if state["foo"] == "bar":
#         return Command(update={"foo": "baz"}, goto="my_other_node")
# whereas conditional edges only route between nodes conditionally without updating the state.
# Navigate to a node in a parent graph (a different node in the parent graph)
# def my_node(state: State) -> Command[Literal["other_subgraph"]]:
#     return Command(
#         update={"foo": "bar"},
#         goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
#         graph=Command.PARENT
#     )




# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")


# specify the START and END nodes (langgraph.graph.START or END)
# add an entry point to tell the graph where to start its work each time
graph_builder.add_edge(START, "chatbot")
# add an exit point to indicate where the graph should finish execution.
graph_builder.add_edge("chatbot", END)



# Graph: define the state > add nodes and edges > compile the graph
# Compile: 1. base checks on the structure 2. add runtime args like checkpointers and breakpoints. 
graph = graph_builder.compile()  # creates a CompiledGraph we can invoke on our state


# Visualize the graph (optional)
# from IPython.display import Image, display # notebook
try:
    # Get the PNG bytes for the Mermaid diagram
    png_data = graph.get_graph().draw_mermaid_png() # mermaid-py library
    # Save the PNG to a file
    with open("mermaid_diagram.png", "wb") as f:
        f.write(png_data)
    print("Mermaid diagram saved as 'mermaid_diagram.png'")
    #display(Image(graph.get_graph().draw_mermaid_png())) # notebook
except Exception:
    # This requires some extra dependencies and is optional
    pass


### Run the chatbot, ask the bot questions
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

# You can exit the chat loop at any time by typing quit, exit, or q.


# Compile the graph with the provided checkpointer
graph = graph_builder.compile(checkpointer=memory)





"""      
Memory 
persistent checkpointing: LangGraph saves the state after each step with a checkpointer and thread_id 

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)
"""

import dotenv
dotenv.load_dotenv() 

from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]

llm = init_chat_model("gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)



### Interact with your chatbot

# Call your chatbot:
user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    # Pick a thread to use as the key for this conversation.
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()

# Ask a follow up question. Note: if you specify another thread_id, or 2 here, the bot has no memory of your name.

user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    {"configurable": {"thread_id": "1"}},
    stream_mode="values",
)

for event in events:
    event["messages"][-1].pretty_print()


# event["messages"][-2]
# 
# HumanMessage(
#     content='Remember my name?', 
#     additional_kwargs={}, 
#     response_metadata={}, 
#     id='f68cf82e-10ff-4034-a366-d171ef384729')



# inspect the state
config = {"configurable": {"thread_id": "1"}}
snapshot = graph.get_state(config)
snapshot.next  # (since the graph ended this turn, `next` is empty. If you fetch a state from within a graph invocation, next tells which node will execute next)



### Add human-in-the-loop controls
# Calling `interrupt`` inside a node will pause execution. Execution can be resumed, together with new input from a human, by passing in a `Command.


import dotenv
dotenv.load_dotenv() 

from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
# Similar to Python's built-in input() function, calling interrupt inside the tool will pause execution. Progress is persisted based on the checkpointer; so if it is persisting with SqliteSaver or PostgresSaver (in-memory checkpointer in this example), it can resume at any time as long as the database is alive. 

tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]

llm = init_chat_model("gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# prompt the chatbox
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

snapshot = graph.get_state(config)
snapshot.next

# resume execution
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()




### Customize state
import dotenv
dotenv.load_dotenv() 

from typing import Annotated
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str

@tool
# Note that because we are generating a ToolMessage for a state update, we generally require the ID of the corresponding tool call. We can use LangChain's InjectedToolCallId to signal that this argument should not be revealed to the model in the tool's schema.
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    return Command(update=state_update)


tool = TavilySearch(max_results=2)
tools = [tool, human_assistance]

llm = init_chat_model("gemini-2.5-flash")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# prompt the chatbot
user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
    )
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# add human assistance
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(
    human_command,
    config,
    stream_mode="values"
    )
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


snapshot = graph.get_state(config)
{k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}

# at any point (including when interrupted), you can manually override a key using
graph.update_state(config, {"name": "LangGraph (library)"})
snapshot = graph.get_state(config)
{k: v for k, v in snapshot.values.items() if k in ("name", "birthday")}
# confirm the new value been reflected




### built-in time travel functionality



config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm learning LangGraph. "
                    "Could you do some research on it for me?"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Ya that's helpful. Maybe I'll "
                    "build an autonomous agent with it!"
                ),
            },
        ],
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()


# Now that you have added steps to the chatbot, you can replay the full state history to see everything that occurred.
to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state
# Checkpoints are saved for every step of the graph. This spans invocations so you can rewind across a full thread's history.

# Resume from the to_replay state.
print(to_replay.next)
print(to_replay.config)

# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer. The checkpoint's to_replay.config contains a checkpoint_id timestamp. Providing this checkpoint_id value tells LangGraph's checkpointer to load the state from that moment in time.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()





from langchain_core.runnables import RunnableLambda

def add_one(x: int) -> int:
    return x + 1

runnable = RunnableLambda(add_one)

runnable.invoke(1) # returns 2
runnable.batch([1, 2, 3]) # returns [2, 3, 4]


