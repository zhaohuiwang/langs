
""""
NOTE:
Functional API vs. Graph API
1. Visualization: The Graph API makes it easy to visualize the workflow as a graph which can be useful for debugging, understanding the workflow, and sharing with others. The Functional API does not support visualization as the graph is dynamically generated during runtime.
2. Control flow: The Functional API does not require thinking about graph structure. You can use standard Python constructs to define workflows. This will usually trim the amount of code you need to write.
3. Short-term memory: The GraphAPI requires declaring a State and may require defining reducers to manage updates to the graph state. @entrypoint and @tasks do not require explicit state management as their state is scoped to the function and is not shared across functions.
4. Checkpointing: Both APIs generate and use checkpoints. In the Graph API a new checkpoint is generated after every superstep. In the Functional API, when tasks are executed, their results are saved to an existing checkpoint associated with the given entrypoint instead of creating a new checkpoint.


The Functional API allows you to add LangGraph's key features — persistence, memory, human-in-the-loop, and streaming — to your applications with minimal changes to your existing code.

The Functional API uses two key building blocks:

@entrypoint -  used to create a workflow from a function. It encapsulates workflow logic and manages execution flow, including handling long-running tasks and interrupts. The function must accept a single POSITIONAL argument (or a dictionary with multiple piece of date) as the workflow input. Using the @entrypoint yields a Pregal object that can be executed using the invoke, ainvoke, stream and astream methods.

@task - Represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously within an entrypoint. Tasks return a future-like object that can be awaited or resolved synchronously. To obtain the result of a task, call the .result() method, or await keyword in async function. 

NOTE: LangGraph, built on JavaScript/TypeScript, often leverages async/await for managing asynchronous workflows, such as fetching data, processing nodes, or handling I/O operations in a graph-based structure. In a LangGraph async function, the await keyword is used to pause the execution of the function until a Promise (or an asynchronous operation) resolves, allowing the function to handle asynchronous tasks in a synchronous-like manner. 

NOTE: Routine (regular function) vs Coroutine
Routine: Executes synchronously and sequentially. Once called, it runs to completion before returning control to the caller. It is defined using def. It returns a value directly upon completion.
Coroutine: Executes asynchronously and cooperatively. It can be suspended at await expressions, allowing other coroutines to run, and then resumed later from the point of suspension. It is defined using async def and typically uses await to pause execution for I/O operation or other asynchronous tasks. When called, it immdeiately returns a coroutine object. The actual resulst of coroutine is obtained by "awaiting" it within another coroutine or by running it using an event loop.


NOTE:
Pregel implements LangGraph's runtime, managing the execution of LangGraph applications. Pregel runtime is named after Google's Pregal algorithm, which is an efficient method for large-scale parallel computation using graphs.

A pregal instance can be created either by compiling a StateGraph or by constructing an entrypoint, and be invoked with input.
In LangGraph, Pregel combines actors (PregalNode, runnable interface) and channels into a single application. Actors subscribe to channels, read data from channels and write data to channels. 

Key Methods of the Pregel Object
The Pregel object provides several methods for interacting with the graph:

- invoke(input, config): Runs the graph with a single input and returns the output.
- stream(input, config): Streams intermediate results for each step, useful for debugging or real-time updates.
- get_state(): Retrieves the current state of the graph.
- update_state(values): Updates the graph's state manually.
- get_graph(): Returns a drawable representation of the computation graph.
- with_config(config): Creates a copy of the Pregel object with an updated configuration (e.g., setting recursion_limit).
- get_state(): Retrieves the current thread state (stored by the checkpoint).
- get_state_history(config): Retrive the history of the thread (checkpoints). e.g. list(graph_name.get_state_history(config))

"""


import time
from typing import Callable, TypedDict, Union
import uuid

from langchain_core.messages import BaseMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool, tool as create_tool
from langchain_core.runnables import RunnableConfig

from langgraph.cache.memory import InMemoryCache
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer 

from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from langgraph.prebuilt.interrupt import HumanInterruptConfig, HumanInterrupt
from langgraph.prebuilt import create_react_agent

from langgraph.types import Command, interrupt, RetryPolicy, CachePolicy


import dotenv
dotenv.load_dotenv() 

# Initialize the LLM model
llm = init_chat_model("gemini-2.5-flash")


### Parallel execution
# Examle 1: simple math to add a constant integer to a list of intergers
@task
def add_one(base: int, number: int) -> int:
    return base + number

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

@entrypoint(checkpointer=checkpointer)
def graph(numbers: list[int]) -> list[str]:
    futures = [add_one(5, i) for i in numbers]
    return [f.result() for f in futures]

graph.invoke([1,2,3], config=config)

# Example 2: using llm to generate a list of essays  
# Task that generates a paragraph about a given topic
@task
def generate_paragraph(topic: str) -> str:
    response = llm.invoke([
        {"role": "system", "content": "You are a helpful assistant that writes educational paragraphs."},
        {"role": "user", "content": f"Write a paragraph about {topic}."}
    ])
    return response.content

# Create a checkpointer for persistence
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(topics: list[str]) -> str:
    """Generates multiple paragraphs in parallel (by invoking them concurrently) and combines them."""
    futures = [generate_paragraph(topic) for topic in topics]
    paragraphs = [f.result() for f in futures]
    return "\n\n".join(paragraphs)

# Run the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
result = workflow.invoke(["quantum computing", "climate change", "history of aviation", "the history of flight"], config=config)
print(result)



### The Functional API and the Graph API can be used together in the same application as they share the same underlying runtime.


# Define the shared state type
class State(TypedDict):
    foo: int

# Define a simple transformation node
def double(state: State) -> State:
    return {"foo": state["foo"] * 2}

# Build the graph using the Graph API
builder = (
    StateGraph(State)
    .add_node("double", double)
    .set_entry_point("double")
    )
graph = builder.compile()

# Define the functional API workflow
checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def workflow(x: int) -> dict:
    result = graph.invoke({"foo": x})
    return {"bar": result["foo"]}

# Execute the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(workflow.invoke(5, config=config))  # Output: {'bar': 10}

### Call another entrypoint from within an entrypoint or task

# Initialize a checkpointer
checkpointer = InMemorySaver()

# A reusable sub-workflow that multiplies a number
@entrypoint()# Will automatically use the checkpointer from the parent entrypoint
def multiply(inputs: dict) -> int:
    return inputs["a"] * inputs["b"]

# Main workflow that invokes the sub-workflow
@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> dict:
    result = multiply.invoke({"a": inputs["x"], "b": inputs["y"]})
    return {"product": result}

# Execute the main workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
print(main.invoke({"x": 6, "y": 7}, config=config))  # Output: {'product': 42}




checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def main(inputs: dict) -> int:
    writer = get_stream_writer() 
    writer("Started processing") 
    result = inputs["x"] * 2
    writer(f"Result is {result}") 
    return result

config = {"configurable": {"thread_id": "abc"}}

pregel_stream_generator = main.stream(
    {"x": 5},
    stream_mode=["custom", "updates"], 
    config=config
    )

print(list(pregel_stream_generator))
# NOTE: a python generator can only be iterated once. Once a generator completes its iteration, it is considered exhausted. Any further attempts to retrive values from it (via next() or a for loop) will continue raise StopIteration. The generator itself still exists in the memory, but it can not produce additional values unless recreated.
# here both list or for loop exhause the generator.
for mode, chunk in pregel_stream_generator:
    print(f"{mode}: {chunk}")

### Retry policy
# Configure the RetryPolicy to retry on ValueError. The default RetryPolicy is optimized for retrying specific network errors.
retry_policy = RetryPolicy(retry_on=ValueError)

@task(retry_policy=retry_policy)
def function_here():
    pass



### Caching Tasks

@task(cache_policy=CachePolicy(ttl=5))  # time to live in second
def slow_add(x: int) -> int:
    time.sleep(10)
    return x * 2

@entrypoint(cache=InMemoryCache())
def main(inputs: dict) -> dict[str, int]:
    result1 = slow_add(inputs["x"]).result()
    result2 = slow_add(inputs["x"]).result()
    return {"result1": result1, "result2": result2}


for chunk in main.stream({"x": 5}, stream_mode="updates"):
    print(chunk)


### human-in-the-loop workflow
@task
def step_1(input_query):
    """Append bar."""
    return f"{input_query} bar"

@task
def human_feedback(input_query):
    """Append user input."""
    # interrupt() is called inside a task, enabling a human to review and edit the output of the previous task. 
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"

@task
def step_3(input_query):
    """Append qux."""
    return f"{input_query} qux"

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def graph(input_query):
    result_1 = step_1(input_query).result()
    result_2 = human_feedback(result_1).result()
    result_3 = step_3(result_2).result()

    return result_3

config = {"configurable": {"thread_id": "1"}}

for event in graph.stream("foo", config):
    print(event)
    print("\n")

# list(graph.stream("foo", config))

# [
#     {'step_1': 'foo bar'},
#     {'__interrupt__': (Interrupt(value='Please provide feedback: foo bar', id='72068c7561377a42e998a6a0d7c38b37'),)}
#     ]

# Continue execution - we issue a Command containing the data expected by the human_feedback task
for event in graph.stream(Command(resume="baz"), config):
    print(event)
    print("\n")

# list(graph.stream(Command(resume="baz"), config))
# [
#     {'human_feedback': 'foo bar baz'},
#     {'step_3': 'foo bar baz qux'}, {'graph': 'foo bar baz qux'}
#     ]


# a review_tool_call function that calls interrupt. When this function is called, execution will be paused until we issue a command to resume it.

def add_human_in_the_loop(
    tool: Callable | BaseTool,
    *,
    interrupt_config: HumanInterruptConfig = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    if interrupt_config is None:
        interrupt_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }

    @create_tool(  
        tool.name,
        description=tool.description,
        args_schema=tool.args_schema
    )
    def call_tool_with_interrupt(config: RunnableConfig, **tool_input):
        request: HumanInterrupt = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "config": interrupt_config,
            "description": "Please review the tool call"
        }
        response = interrupt([request])[0]  
        # approve the tool call
        if response["type"] == "accept":
            tool_response = tool.invoke(tool_input, config)
        # update tool call args
        elif response["type"] == "edit":
            tool_input = response["args"]["args"]
            tool_response = tool.invoke(tool_input, config)
        # respond to the LLM with user feedback
        elif response["type"] == "response":
            user_feedback = response["args"]
            tool_response = user_feedback
        else:
            raise ValueError(f"Unsupported interrupt response type: {response['type']}")

        return tool_response

    return call_tool_with_interrupt

checkpointer = InMemorySaver()

def book_hotel(hotel_name: str):
   """Book a hotel"""
   return f"Successfully booked a stay at {hotel_name}."


agent = create_react_agent(
    model=llm,
    tools=[
        add_human_in_the_loop(book_hotel), 
    ],
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "1"}}

# Run the agent
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "book a stay at McKittrick hotel"}]},
    config
):
    print(chunk)
    print("\n")




from typing import Optional
from langgraph.func import entrypoint
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

@entrypoint(checkpointer=checkpointer)
def accumulate(n: int, *, previous: Optional[int]) -> entrypoint.final[int, int]:
    previous = previous or 0
    total = previous + n
    # Return the *previous* value to the caller but save the *new* total to the checkpoint.
    return entrypoint.final(value=previous, save=total)

config = {"configurable": {"thread_id": "my-thread"}}

print(accumulate.invoke(1, config=config))  # 0
print(accumulate.invoke(2, config=config))  # 1
print(accumulate.invoke(3, config=config))  # 3

list(accumulate.get_state_history(config))


# The InMemorySaver stores checkpoints in an internal dictionary, where the keys are thread_id values. To reveal all checkpointer threads
for thread_key, state in checkpointer.storage.items():
    thread_id = thread_key 
    print(f"Thread ID: {thread_id}\n State: {state}")

# to erase all checkpointer threads
checkpointer.storage.clear()

# To verify the python top level thread - MainThread
import threading
for thread in threading.enumerate():
    print(f"Thread name: {thread.name}, ID: {thread.ident}, Alive: {thread.is_alive()}")
# Returns: Thread name: MainThread, ID: 139741804627776, Alive: True
# The `MainThread`` is the thread that starts automatically when you run a Python program. The MainThread executes the top-level code of your script. It responsible for starting other threads if your program creates them.



### Example to illustrate coroutine and asynchronous execution
import asyncio

# Define a coroutine
async def say_hello():
    print("Hello")
    await asyncio.sleep(1)  # Simulate an I/O-bound task (e.g., waiting for data)
    print("World")

# Define another coroutine
async def say_goodbye():
    print("Good")
    await asyncio.sleep(0.5)  # Simulate a shorter I/O-bound task
    print("Bye")

# Main coroutine to run multiple coroutines
async def main():
    # Run coroutines concurrently
    await asyncio.gather(say_hello(), say_goodbye())

# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())

# Returs:
#   Hello
#   Good
#   Bye
#   World
"""
NOTE: Key Points
Use async def to define a coroutine and await to pause it for asynchronous tasks.
Use asyncio.run() to execute the main coroutine in an event loop.
Use asyncio.gather() to run multiple coroutines concurrently.
Coroutines are best for I/O-bound tasks, not CPU-bound tasks (use multiprocessing or concurrent.futures for CPU-intensive work).
Always run await inside an async def function; you cannot use await in regular functions.

"""