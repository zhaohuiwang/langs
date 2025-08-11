
""""
NOTE:
State Key Points
The state is a mutable, shared data structure that persists throughout the graph's execution.
It can be a simple dictionary, a typed dictionary (TypedDict), a Pydantic model, or a custom class.
The state is updated by nodes, and LangGraph handles the mechanics of passing and updating the state between nodes.
You can customize the state structure to fit your application's needs, making it as simple or complex as required.

NOTE: Conditional edges are evaluated at runtime immediately after the source node finishes executing.
Syntax: add_conditional_edges(source:str, path:Callable[...] | Runnable[...], path_map=dict[Hashable, str] | list[str] | None) -> Self
The core of evaluation is the `path` callable/runnable. It inspects the state and applied your custom condition(s) for return.

Comparison of graph.invoke() and graph.stream() in LangGraph
(ainvoke and sstream methods for asynchronous execution)
purpose, output and use case:
graph.invoke(): Executes the graph and returns the final output after the entire process completes. Returns a single result (e.g., the final state or output value). Suitable for simple workflows where you only need the final result (e.g., a single response).

graph.stream(): Executes the graph and streams intermediate outputs or events in real time as the process runs. Yields a sequence of events (e.g., node outputs, state updates) that you can iterate over. Ideal for real-time applications (e.g., chatbots) or debugging, where you want to process or display partial results.

Execution Style:
graph.invoke(): Synchronous and blocking; waits for the full computation before returning.
graph.stream(): Asynchronous and non-blocking; provides incremental results as they become available.

Parameters:
Both accept input and config (e.g., {"configurable": {"thread_id": "123"}} for stateful graphs).
graph.stream() supports an additional stream_mode parameter (e.g., "values", "updates", "debug") to control what is streamed.


"""


from utils import draw_save_mermaid_png

import operator
import random

from dataclasses import dataclass
from pydantic import BaseModel

from typing import Annotated, Any, List, Literal, Optional, Sequence, TypedDict

from langgraph.cache.memory import InMemoryCache
from langgraph.errors import GraphRecursionError

from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, Command, RetryPolicy,Send
from langgraph.graph import StateGraph, START, END

import dotenv
dotenv.load_dotenv() 

llm = init_chat_model("gemini-2.5-flash")


class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int

def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}

graph = StateGraph(State).add_node(node).set_entry_point("node").compile()

result = graph.invoke({"messages": [HumanMessage("Hi")]})
for message in result["messages"]:
    message.pretty_print() # pretty_print() method is a utility function.
draw_save_mermaid_png(graph)
# ============

# Hi
# ================================== Ai Message ==================================

# Hello!

result
# {'messages': [
#     HumanMessage(content='Hi', additional_kwargs={}, response_metadata={}),
#     AIMessage(content='Hello!', additional_kwargs={}, response_metadata={})],
#     'extra_field': 10
#     }
# Note: key 'messages' is a list of test messages




### Process state updates with reducers - Each key in the state can have its own independent reducer function.
# A reducer is typically a function that takes the current state and a state update (new data from a node or input) and returns the updated state. It ensures that state changes are consistent, deterministic, and aligned with the workflow’s requirements, avoiding conflicts or unintended overwrites. Types of Reducers: 1. default reducer (overwriting existing keys with new values or merging dictionaries) 2. Custom Reducers. Reducers are typically specified when defining the graph's schema or channels using the StateGraph class and Annotated types with a reducer function.


def add(left, right):
    """Can also import `add` from the `operator` built-in."""
    return left + right

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add] # reducer for appending messages
    extra_field: int


# there is even a pre-build MessageState for convenience
class State(MessagesState):
    extra_field: int


def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}


### input and output schemas - By default, StateGraph operates with a single schema, and all nodes are expected to communicate using that schema. However it is possible to define distinct input and output schemas for a graph. output schema - filters the internal data to return only the relevant information. input schema - ensures that the provided input matches the expected structure.

class InputState(TypedDict): # input schema
    question: str

class OutputState(TypedDict): # output schema
    answer: str

class OverallState(InputState, OutputState): # overall schema
    pass

# Define the node that processes the input and generates an answer
def answer_node(state: InputState):
    # Example answer and an extra key
    return {"answer": "bye", "question": state["question"]}

# Build the graph with input and output schemas specified
graph = (
    StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
    .add_node(answer_node)
    .add_edge(START, "answer_node")
    .add_edge("answer_node", END)
    .compile()
    )

# Invoke the graph with an input and print the result
print(graph.invoke({"question": "hi"})) 
# {'answer': 'bye'}
# Reason: only the answer key in specified in the OutputState schema.

draw_save_mermaid_png(graph)

### pass private state between nodes

# The overall state of the graph (this is the public state shared across nodes)
class OverallState(TypedDict):
    a: str

# Output from node_1 contains private data that is not part of the overall state
class Node1Output(TypedDict):
    private_data: str

# Node 2 input only requests the private data available after node_1
class Node2Input(TypedDict):
    private_data: str
    
# The private data is only shared between node_1 and node_2
def node_1(state: OverallState) -> Node1Output:
    output = {"private_data": "set by node_1"}
    print(f"Entered node `node_1`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

def node_2(state: Node2Input) -> OverallState:
    output = {"a": "set by node_2"}
    print(f"Entered node `node_2`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Node 3 only has access to the overall state (no access to private data from node_1)
def node_3(state: OverallState) -> OverallState:
    output = {"a": "set by node_3"}
    print(f"Entered node `node_3`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Connect nodes in a sequence, node_2 accepts private data from node_1, whereas node_3 does not see the private data.
graph = (
    # add_sequence() instead of add_node()
    StateGraph(OverallState).add_sequence([node_1, node_2, node_3])
    .add_edge(START, "node_1")
    .compile()
    )

# Invoke the graph with the initial state
response = graph.invoke(
    {"a": "set at start",}
    )

# Entered node `node_1`:
#         Input: {'a': 'set at start'}.
#         Returned: {'private_data': 'set by node_1'}
# Entered node `node_2`:
#         Input: {'private_data': 'set by node_1'}.
#         Returned: {'a': 'set by node_2'}
# Entered node `node_3`:
#         Input: {'a': 'set by node_2'}.
#         Returned: {'a': 'set by node_3'}

print()
print(f"Output of graph invocation: {response}")
# Output of graph invocation: {'a': 'set by node_3'}



### Use Pydantic models for graph state
# dataclasses.dataclass and typing.TypedDict perform static checking only and mybe easy.  Pydantic.BaseModel is versatile

class ChatState(BaseModel):
    messages: List[AnyMessage]
    context: str
# should use AnyMessage (rather than BaseMessage) for proper serialization/deserialization when using message objects over the wire.

def add_message(state: ChatState):
    return {"messages": state.messages + [AIMessage(content="Hello there!")]}

graph = (
    StateGraph(ChatState)
    .add_node("add_message", add_message)
    .add_edge(START, "add_message")
    .add_edge("add_message", END)
    .compile()
    )

# Create input with a message
initial_state = ChatState(
    messages=[HumanMessage(content="Hi")], context="Customer support chat"
)

result = graph.invoke(initial_state)
result

# {
#   'messages': [
#       HumanMessage(content='Hi', additional_kwargs={}, response_metadata={}),
#       AIMessage(content='Hello there!', additional_kwargs={},response_metadata={})
#       ],
#   'context': 'Customer support chat'
#   }

draw_save_mermaid_png(graph)

# Convert back to Pydantic model to see message types
output_model = ChatState(**result)
for i, msg in enumerate(output_model.messages):
    print(f"Message {i}: {type(msg).__name__} - {msg.content}")




# coerce or validation error

class CoercionExample(BaseModel):
    number: int # string numbers will be coerced to integers but not letters
    flag: bool  # string booleans will be parsed to bool

def inspect_node(state: CoercionExample):
    print(f"number: {state.number} (type: {type(state.number)})")
    print(f"flag: {state.flag} (type: {type(state.flag)})")
    return {}

graph = (
    StateGraph(CoercionExample)
    .add_node("inspect", inspect_node)
    .add_edge(START, "inspect")
    .add_edge("inspect", END)
    .compile()
    )

# Demonstrate coercion with string inputs that will be converted
result = graph.invoke({"number": "42", "flag": "true"})
# This would fail with a validation error
try:
    graph.invoke({"number": "not-a-number", "flag": "true"})
except Exception as e:
    print(f"\nExpected validation error: {e}")

draw_save_mermaid_png(graph)


### Add runtime configuration - decide a LLM at runtime

@dataclass
class ContextSchema:
    model_provider: str = "openai"
    system_message: str | None = None

MODELS = {
    "openai": init_chat_model("openai:gpt-4.1-mini"),
    "gemini": init_chat_model("google_vertexai:gemini-2.5-flash"),
}
# init_chat_model(provider:model_name) provider is optional

def call_model(state: MessagesState, runtime: Runtime[ContextSchema]):
    model = MODELS[runtime.context.model_provider]
    messages = state["messages"]
    if (system_message := runtime.context.system_message):
        messages = [SystemMessage(system_message)] + messages
    response = model.invoke(messages)
    return {"messages": [response]}

graph = (
    StateGraph(MessagesState, context_schema=ContextSchema)
    .add_node("model", call_model)
    .add_edge(START, "model")
    .add_edge("model", END)
    .compile()
    )

# Usage
input_message = {"role": "user", "content": "hi"}
response = graph.invoke({"messages": [input_message]}, context={"model_provider": "gemini", "system_message": "Respond in Italian."})
for message in response["messages"]:
    message.pretty_print()

draw_save_mermaid_png(graph)


### retry policy - parameter takes in a RetryPolicy named tuple object

graph = (
    StateGraph(MessagesState)
    .add_node(
        "node_name",
        node_function,
        retry_policy=RetryPolicy(),
        )
    .add_node(
        "query_database",
        query_database,
        retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError),
        )
    .add_node("model", call_model, retry_policy=RetryPolicy(max_attempts=5))
    .add_edge(START, "model")
    .add_edge("model", "query_database")
    .add_edge("query_database", END)
    .compile()
    )


# use case retrying an API call
# Define the state structure
class WeatherState(TypedDict):
    city: str
    weather_data: dict
    error: str

# Simulate an API call that may fail
def fetch_weather_api(city: str) -> dict:
    # Simulate random failures (30% chance of failure)
    if random.random() < 0.3:
        raise ValueError("API call failed: Network issue or rate limit")
    return {"city": city, "temperature": 25 + random.randint(-5, 5), "condition": "sunny"}

# Node that fetches weather data
def fetch_weather_node(state: WeatherState) -> WeatherState:
    try:
        weather_data = fetch_weather_api(state["city"])
        return {"weather_data": weather_data, "error": None}
    except ValueError as e:
        return {"weather_data": None, "error": str(e)}

# Node to process the weather data
def process_weather_node(state: WeatherState) -> WeatherState:
    if state["error"]:
        return {"weather_data": {"message": f"Failed to fetch weather: {state['error']}"}}
    data = state["weather_data"]
    return {
        "weather_data": {
            "message": f"Weather in {data['city']}: {data['temperature']}°C, {data['condition']}"
        }
    }

# Create the graph workflow
workflow = (
    StateGraph(WeatherState)
    .add_node(
        "fetch_weather",
        fetch_weather_node,
        retry=RetryPolicy(
            max_attempts=3,  # Retry up to 3 times
            backoff_factor=2.0,  # Exponential backoff: 1s, 2s, 4s
            retry_on=(ValueError)  # Retry only on ValueError
            )
        )
    .add_node("process_weather", process_weather_node)
    .add_edge("fetch_weather", "process_weather")
    .set_entry_point("fetch_weather")
    .set_finish_point("process_weather")
)
# Compile the graph
app = workflow.compile()

# Run the graph
input_state = {"city": "New York", "weather_data": None, "error": None}
result = app.invoke(input_state)
print(result["weather_data"])

### Add node caching
"""NOTE:
Checkpointers as the Core Caching Mechanism: 
Role: Checkpointers like MemorySaver or SQLiteSaver save the graph's state (a dictionary, TypedDict, or Pydantic model) at each step, keyed by a thread_id in the config dictionary (e.g., {"configurable": {"thread_id": "123"}}).
Behavior: When a graph is executed with a previously used thread_id, the checkpointer retrieves the cached state, potentially skipping node execution if the state is complete.
Types: 1. MemorySaver: Stores states in memory, reset on program restart. 2. SQLiteSaver: Persists states in a SQLite database, suitable for production.
"""
builder=(
    StateGraph(MessagesState)
    .add_node(
        "node_name",
        node_function,
        cache_policy=CachePolicy(ttl=120), # a time to live of 120 seconds
    )
)
graph = builder.compile(cache=InMemoryCache())

draw_save_mermaid_png(graph)

### Create a sequence of steps
# two built-in short-hand: .add_node() and .add_sequence() 
# .add_sequence([a,b,c,d]), creates (the default) a strictly sequential flow, like START → a → b → c → d → END. You can still add branches though.

# .add_node(a).add_node(b).add_node(c).add_node(d) There is no default and it gives you flexibility to create any simple or complex graph with branching and convergence.


# def step_1():
#    pass
# builder.add_node(step_1)
# .add_edge takes the names of nodes (function), which for functions defaults to node.__name__ (function.__name__ or "step_1" here). We can specify a custom name, for example,

# builder.add_node("my_node", step_1)

# the names are then being use in .add_edge() and displayed in the graph image.


### Create branches
# parallelization through fan-out and fan-in mechanisms 
### Defer node execution until all other pending tasks are completed.


class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def b_2(state: State):
    print(f'Adding "B_2" to {state["aggregate"]}')
    return {"aggregate": ["B_2"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}


builder = (
    StateGraph(State)
    #.add_sequence([a, b, b_2, c, d])
    .add_node(a)
    .add_node(b)
    .add_node(b_2)
    .add_node(c)
    .add_node(d, defer=True) # Defer node execution until all other pending tasks are completed, three banches with various lengthes
    .add_edge(START, "a")
    .add_edge("a", "b")
    .add_edge("a", "c")
    .add_edge("b", "b_2")
    .add_edge("b_2", "d")
    .add_edge("c", "d")
    .add_edge("d", END)
)
graph = builder.compile()

# With the reducer, you can see that the values added in each node are accumulated.
graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
draw_save_mermaid_png(graph)

# add_node() creates flow graph `START → a → (b, b → b_2 → c, c) → d → END`
# add_sequence() creates one additional linear flow graph `START → a → (b → b_2 → c, b → b_2, c)→ d → END` 


### Conditional branching - If your fan-out should vary at runtime based on the state, you can use add_conditional_edges to select one or more paths using the graph state.

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    # Add a key to the state. We will set this key to determine
    # how we branch.
    which: str

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"], "which": "c"}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

builder = (
    StateGraph(State)
    .add_node(a)
    .add_node(b)
    .add_node(c)
    .add_edge(START, "a")
    .add_edge("b", END)
    .add_edge("c", END)
    )

def conditional_edge(state: State) -> Literal["b", "c"]:
    # Fill in arbitrary logic here that uses the state to determine the next node
    return state["which"]

# Example to show conditional edges route to multiple destination nodes.
def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]

# case I: single destination mode
builder.add_conditional_edges("a", conditional_edge)
# case II: muitile destination node
builder.add_conditional_edges("a", route_bc_or_cd)

graph = builder.compile()

result = graph.invoke({"aggregate": []})
print(result)

"""

"""

### Send API for Map-reduce and other advanced branching patterns

class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str

def generate_topics(state: OverallState):
    return {"subjects": ["lions", "elephants", "penguins"]}

def generate_joke(state: OverallState):
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
    }
    return {"jokes": [joke_map[state["subject"]]]}

def continue_to_jokes(state: OverallState):
    if state["subjects"] is None:
        return 'END'
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def best_joke(state: OverallState):
    return {"best_selected_joke": "penguins"}

builder = (
    StateGraph(OverallState)
    .add_node("generate_topics", generate_topics)
    .add_node("generate_joke", generate_joke)
    .add_node("best_joke", best_joke)
    .add_edge(START, "generate_topics")
    .add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
    .add_edge("generate_joke", "best_joke")
    .add_edge("best_joke", END)
    .add_edge("generate_topics", END)
    )
graph = builder.compile()

draw_save_mermaid_png(graph)

graph.invoke({"topic": "animals"})

{
    'topic': 'animals',
    'subjects': ['lions', 'elephants', 'penguins'],
    'jokes': ["Why don't lions like fast food? Because they can't catch it!", "Why don't elephants use computers? They're afraid of the mouse!", "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."],
    'best_selected_joke': 'penguins'
 }

# Call the graph: here we call it to generate a list of jokes
for step in graph.stream({"topic": "animals"}):
    print(step)

"""
NOTE: Conditional edges are evaluated at runtime immediately after the source node finishes executing.
Syntax: add_conditional_edges(source:str, path:Callable[...] | Runnable[...], path_map=dict[Hashable, str] | list[str] | None) -> Self
The core of evaluation is the `path` callable/runnable. It inspects the state and applied your custom condition(s) for return.

"""

### Create and control loops

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# conditional edge
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder = (
    StateGraph(State)
    .add_node(a)
    .add_node(b)
    .add_edge(START, "a")
    .add_conditional_edges("a", route)
    .add_edge("b", "a")
    )
graph = builder.compile()

graph.invoke({"aggregate": []})

draw_save_mermaid_png(graph)

try:
    graph.invoke({"aggregate": []}, {"recursion_limit": 4})
except GraphRecursionError:
    print("Recursion Error")

### Async
# Because many LangChain objects implement the Runnable Protocol which has async variants of all the sync methods it's typically fairly quick to upgrade a sync graph to an async graph.
# To convert a sync implementation of the graph to an async implementation, you will need to:
# Update nodes use async def instead of def.
# Update the code inside to use await appropriately.
# Invoke the graph with .ainvoke or .astream as desired.

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph

async def node(state: MessagesState): 
    new_message = await llm.ainvoke(state["messages"]) 
    return {"messages": [new_message]}

builder = StateGraph(MessagesState).add_node(node).set_entry_point("node")
graph = builder.compile()

input_message = {"role": "user", "content": "Hello"}
result = await graph.ainvoke({"messages": [input_message]})
# SyntaxError: 'await' outside function


# Run the async function
import asyncio
async def run_graph():
    input_message = {"role": "user", "content": "Hello"}
    result = await graph.ainvoke({"messages": [input_message]})
    print(result)

# Run the async function
asyncio.run(run_graph())

### Command - Combine control flow and state updates
 
class State(TypedDict):
    foo: Annotated[str, operator.add]# NOTE: we define a reducer here

def node_a(state: State):
    print("Called A")
    value = random.choice(["a", "b"])
    # this is a replacement for a conditional edge function
    goto = "node_b" if value == "a" else "node_c"
    
    # Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        update={"foo": value},
        goto=goto, # navigate to node_b or node_c in the parent graph
        graph=Command.PARENT, # navigate to the closest parent graph relative to the subgraph
    )

subgraph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()

def node_b(state: State):
    print("Called B")
    # NOTE: since we've defined a reducer, we don't need to manually append new characters to existing 'foo' value. instead, reducer will append these automatically (via operator.add)
    return {"foo": "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": "c"}

builder = (
    StateGraph(State)
    .add_edge(START, "subgraph")
    .add_node("subgraph", subgraph)
    .add_node(node_b)
    .add_node(node_c)
    )

graph = builder.compile()
graph.invoke({"foo": ""})
draw_save_mermaid_png(graph)

"""
NOTE: A subgraph is a self-contained portion of a larger graph (parent) that encapsulates a specific set of nodes and edges. It represents a modular or reusable component of the workflow. Subgraphs can be treated as a single node within the larger graph, allowing for hierarchical structuring and reusability.

"""
