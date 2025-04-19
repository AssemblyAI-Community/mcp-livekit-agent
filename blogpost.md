# How to build a voice agent with OpenAI and LiveKit Agents


[LiveKit](https://livekit.io/) is a powerful open-source platform for building real-time audio and video applications. Building upon WebRTC, it simplifies the complexities of real-time communication. Recently, LiveKit introduced the [Agents Framework](https://github.com/livekit/agents), enabling developers to integrate AI agents directly into their real-time sessions. These agents can process media streams, interact with external services, and add sophisticated AI capabilities to applications.

One exciting application of LiveKit Agents is building **voice agents** – autonomous systems that interact with users through natural language conversation. These agents typically follow a Speech-to-Text (STT) -> Large Language Model (LLM) -> Text-to-Speech (TTS) pipeline to understand user input, determine an appropriate response or action, and communicate back verbally.

In a [previous post](https://www.assemblyai.com/blog/livekit-realtime-speech-to-text), we explored how to integrate real-time Speech-to-Text into a LiveKit application using AssemblyAI. Now, we'll take it a step further and build a complete voice agent that leverages:

*   **LiveKit Agents:** For the real-time infrastructure and agent framework.
*   **AssemblyAI:** For fast and accurate real-time Speech-to-Text.
*   **OpenAI:** For powerful language understanding and generation (LLM).
*   **(Optional) TTS service (e.g., Silero, OpenAI TTS):** To give the agent a voice.

We'll cover setting up the project, processing audio streams, interacting with an LLM using LiveKit's function calling capabilities (built upon the Model Context Protocol), and triggering actions based on the conversation.

## Prerequisites

Before we start, make sure you have the following:

*   **Python environment:** (Version 3.9 or later recommended)
*   **LiveKit Account:** Sign up at [livekit.io](https://livekit.io/) (Cloud account is free for small projects).
*   **AssemblyAI API Key:** Get one from the [AssemblyAI dashboard](https://www.assemblyai.com/dashboard/signup). Note that Streaming STT requires adding a payment method.
*   **OpenAI API Key:** Obtain one from the [OpenAI platform](https://platform.openai.com/).
*   **Basic understanding of LiveKit:** Familiarity with concepts like Rooms, Participants, and Tracks will be helpful (take a look at our [previous post](https://www.assemblyai.com/blog/livekit-realtime-speech-to-text) or [LiveKit Docs](https://docs.livekit.io/)).

## Setting up the Project

Building a LiveKit application with an agent involves three main components:

1.  **LiveKit Server:** The central hub managing real-time connections and data flow.
2.  **Frontend Application:** The user interface for interaction.
3.  **AI Agent:** The backend process performing the STT, LLM, and TTS tasks.

For the server and frontend, we can reuse the setup from our previous STT tutorial using **LiveKit Cloud** and the **LiveKit Agents Playground**.

**1. Set up LiveKit Cloud:**
   *   Follow the steps in this [previous article](https://www.assemblyai.com/blog/livekit-realtime-speech-to-text/#step-1---set-up-a-livekit-server) to create a LiveKit Cloud project.
   *   Create a `.env` file in your project directory and store your `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`.

   ```dotenv
   # .env
   LIVEKIT_URL=wss://YOUR_PROJECT_URL.livekit.cloud
   LIVEKIT_API_KEY=YOUR_API_KEY
   LIVEKIT_API_SECRET=YOUR_API_SECRET
   ```

**2. Add API Keys to `.env**:
   *   Add your AssemblyAI and OpenAI API keys to the `.env` file.

   ```dotenv
   # .env
   # ... (LiveKit keys)
   ASSEMBLYAI_API_KEY=YOUR_ASSEMBLYAI_KEY
   OPENAI_API_KEY=YOUR_OPENAI_KEY
   ```
   *Remember to keep your `.env` file secure and out of version control.*

**3. Set up Python Environment:**
   *   Create and activate a virtual environment:
     ```bash
     # Mac/Linux
     python3 -m venv venv
     . venv/bin/activate

     # Windows
     # python -m venv venv
     # .\venv\Scripts\activate.bat
     ```
   *   Install necessary libraries:
     ```bash
     pip install #TODO
     # Add TTS libraries if needed, e.g., pip install livekit-plugins-silero
     ```

**4. Use the Agents Playground:**
   *   Navigate to [agents-playground.livekit.io](https://agents-playground.livekit.io/) and connect it to your LiveKit Cloud project as shown in the [previous guide](https://www.assemblyai.com/blog/livekit-realtime-speech-to-text/#step-2---set-up-the-livekit-agents-playground). This will serve as our frontend.

With the basic setup complete, we can now focus on building the core logic of our voice agent.

## Building Your Voice Agent

Let's dive into creating a voice agent that can listen to a user's speech, understand it, respond naturally, and even perform actions based on the conversation.

### How it Works

Our voice agent follows a straightforward but powerful pipeline:

1. **Speech-to-Text (STT):** The user's spoken words are transcribed into text using a service like AssemblyAI.
2. **Language Understanding (LLM):** The transcribed text is processed by a Large Language Model (like OpenAI's gpt-4o) to understand the user's intent and determine how to respond.
3. **Text-to-Speech (TTS):** Finally, the LLM's response is converted back into speech using a TTS service, giving our agent a natural voice.

This modular setup lets you easily plug in different AI services and customize the agent's behavior.

While newer models like `gpt-4o-audio` handle speech directly, offering faster responses, the pipeline we've described gives you more flexibility.

### Declaring the Voice Agent

Let's see how we define this pipeline using LiveKit's agent framework. Below is the simplest setup you can get for a livekit voice agent:

```python
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import assemblyai, openai

from dotenv import load_dotenv
load_dotenv() # Load the environment variables from the .env file

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(
        instructions="You are a friendly voice assistant",  # Instructions define the agent's personality or behavior
    )

    session = AgentSession(
        stt=assemblyai.STT(),  # The STT component handles real-time transcription from speech to text
        llm=openai.LLM(model="gpt-4o"),  # An LLM processes the text to understand and generate responses
        tts=openai.TTS(),  # The TTS component gives the agent a natural-sounding voice
    )

    await session.start(agent=agent, room=ctx.room)  # Connects the agent to the LiveKit room to start interaction
    await session.generate_reply(instructions="Greet the user and ask them about their day")  # Initial greeting and prompt to the user

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

As you can see, this setup is lightweight and flexible. We are now using OpenAI's GPT-4o for the language understanding part, but if you feel like trying a different LLM (maybe something like Claude Sonnet 3.7 or a fine-tuned open-source model) swapping it out is pretty straightforward. Same story with the text-to-speech and speech-to-text components: while OpenAI's TTS does the job, you can easily switch to something like ElevenLabs if you want more natural voices or custom voice cloning. For Speech-to-Text, we load our own AssemblyAI LiveKit plugin, which takes care of the real-time transcription. 


#### Decomposing Our Minimal Example

The `entrypoint` function is the main entry point for our LiveKit agent. It serves as the central coordinator that:

- Connects to the LiveKit room using `ctx.connect()`.
- Initializes the `Agent` and attaches it to the `AgentSession`.
- Starts the session, allowing the agent to interact with users in real-time.

The `Agent` class defines the capabilities and behavior of your voice assistant. Think of it as the "brain" that determines how the agent will interpret and respond to user inputs. You can configure it with specific instructions and equip it with tools that enable particular actions or skills.

The `AgentSession` class manages the real-time interaction between your agent and the LiveKit room. It orchestrates the flow of information between users and agents, handling the streaming audio processing and response generation.

When the session starts with `session.start(agent=agent, room=ctx.room)`, your agent connects to the LiveKit room and begins listening for user input. The `session.generate_reply()` method allows your agent to take the initiative, greeting users as soon as they connect rather than waiting for them to speak first.

While our initial code example is intentionally minimal, each component offers numerous configuration options to enhance your agent's performance. You can customize voices, select specific AI models, or even swap out entire components with alternative providers.

One powerful addition is Voice Activity Detection (VAD), which optimizes your agent by only processing audio when someone is actually speaking. This not only improves response time but also reduces costs by avoiding unnecessary transcription of silence or background noise:

```python
from livekit.plugins import silero

session = AgentSession(
        vad=silero.VAD.load(min_silence_duration = 0.1),  # Adds a VAD component to the pipeline
        stt=assemblyai.STT(word_boost = ["Supabase"]),  # Boosts the confidence of specific words
        llm=openai.LLM(model="gpt-4o-mini", temperature = 0.5),  # You can easily switch to a smaller model
        tts=openai.TTS(voice="alloy")  # Specify a particular voice
    )
```

If you run the first code piece as is, you'll have a fully functional voice agent capable of natural conversation. You can talk with it in the LiveKit Agents Playground. While it can already engage in dialogue, it doesn't yet perform specific actions or tasks. Let's address that next by adding function calling capabilities.

## MCP and The Power of Function Calling

Function calling allows your agent to leverage external tools to complete specific tasks that go beyond simple conversation. When the user says something like "Book a meeting with Sarah for tomorrow at 2 PM," your agent understands that it needs to use a calendar booking tool, extract the relevant parameters (who, when), and execute the action.

Without tools, an agent's capabilities remain limited to its pre-trained knowledge and conversational skills. With tools, your agent becomes truly useful. It gets the ability to check databases, call APIs, control devices, or perform any programmatic action you enable.

Traditionally, developers define the logic behind these tools, specifying how AI models interact with external services. The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), introduced by Anthropic, shifts this responsibility from developers to service providers. MCP offers an open, standardized method for external services to define their interactions with language models. This simplifies integration and ensures consistent tool usage.

In our voice agent application, MCP will ensure the model can effectively select the appropriate external tool, formulate structured requests, and handle responses seamlessly.


### Supabase and MCP

MCP is rapidly gaining traction across the AI industry and major tech corps like Microsoft and OpenAI have shown strong support for it. Many companies are rapidly building their own MCP servers to allow language models to interact directly with their services.

[Supabase](https://supabase.com/), the popular open-source Firebase alternative, recently launched their own official MCP server implementation. This makes their powerful database and authentication services directly available to AI agents through a standardized interface.

Here are some examples of what you can ask your agent to do via Supabase MCP:

**Query data with natural language:**
"Show me all customers who signed up in the last month and have spent over $100."
*The agent translates this to SQL and uses `execute_sql` to retrieve the filtered data.*

**Create or modify database structures:**
"Create a new table called 'products' with columns for ID, name, price, and inventory count."
*The agent generates the appropriate SQL schema and uses `apply_migration` to track this change.*

**Analyze database information:**
"What tables do we have in the 'public' schema, and what are their column structures?"
*The agent uses `list_tables` to retrieve schema information and presents it in a readable format.*

**Manage database extensions:**
"Enable the PostGIS extension so we can store geospatial data."
*The agent checks available extensions with `list_extensions` and applies the necessary changes.*

All these capabilities become available without having to build custom integrations for each AI service you might use.

### Equipping Our Voice Agent with Supabase Tools

Let's enhance our agent by connecting it to Supabase's MCP server. 

First, you'll need a Supabase account and access token. If you don't have one yet, [sign up for Supabase](https://supabase.com/) and create a new project for free. Then you can create an access token [here](https://supabase.com/dashboard/account/tokens).

Add your Supabase access token to your `.env` file:


```dotenv
# .env
# ... existing variables
SUPABASE_ACCESS_TOKEN=your_access_token_here
```

Now, let's enhance our agent code to integrate Supabase's MCP server and expose its powerful database tools to our voice assistant. Below is the complete implementation, which we'll then break down and explain in detail:
```python
from __future__ import annotations
import json, os, inspect
from typing import Any, List, Callable, Optional, get_origin

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import assemblyai, openai, silero
from pydantic_ai.mcp import MCPServerStdio

from dotenv import load_dotenv

load_dotenv()


def _py_type(schema: dict) -> Any:
    """Return typing annotation preserving element type for arrays."""
    t = schema.get("type")
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "object": dict}

    if t in type_map:
        return type_map[t]
    if t == "array":
        return List[_py_type(schema.get("items", {}))]
    return Any


def schema_to_google_docstring(description: str, schema: dict) -> str:
    """
    Generate a Google‑style docstring Args section from a description and a JSON schema.
    """
    props = schema["properties"]
    required = set(schema.get("required", []))
    lines = [description, "Args:"]
    for name, prop in props.items():
        # map JSON‐Schema types to Python
        t = prop["type"]
        if t == "array":
            item_t = prop["items"]["type"]
            py_type = f"List[{item_t.capitalize()}]"
        else:
            py_type = t.capitalize()
        if name not in required:
            py_type = f"Optional[{py_type}]"
        desc = prop.get("description", "")
        lines.append(f"    {name} ({py_type}): {desc}")
    return "\n".join(lines)


async def build_livekit_tools(server) -> List[Callable]:
    """
    Turns every MCP ToolDefinition into a LiveKit function_tool
    """
    tools: List[Callable] = []

    for td in await server.list_tools():
        props = td.parameters_json_schema.get("properties", {})
        required = set(td.parameters_json_schema.get("required", []))

        # capture *all* per‑tool data in default‑arg positions
        def make_proxy(tool_def=td, _props=props, _required=required):
            async def proxy(context: RunContext, **kwargs):
                result = await server.call_tool(tool_def.name, arguments=(kwargs or None))
                txt = result.content[0].text
                try:
                    return json.loads(txt)
                except Exception:
                    return txt

            # clean signature so LiveKit/OpenAI introspection works
            sig_params = [
                inspect.Parameter(
                    "context",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=RunContext,
                )
            ]
            ann = {"context": RunContext}
            for name, schema in _props.items():
                sig_params.append(
                    inspect.Parameter(
                        name,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=_py_type(schema),
                        default=inspect._empty if name in _required else None,
                    )
                )
                ann[name] = _py_type(schema)

            proxy.__signature__ = inspect.Signature(sig_params)
            proxy.__annotations__ = ann
            proxy.__name__ = tool_def.name
            proxy.__doc__ = schema_to_google_docstring(tool_def.description or "", tool_def.parameters_json_schema)
            return function_tool(proxy)

        tools.append(make_proxy())  # factory runs with frozen variables

    return tools

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    server = MCPServerStdio(
        "npx",
        args=[
            "-y",
            "@supabase/mcp-server-supabase@latest",
            "--access-token",
            os.environ["SUPABASE_ACCESS_TOKEN"],
        ],
    )

    await server.__aenter__()

    livekit_tools = await build_livekit_tools(server)
    agent = Agent(
        instructions="You are a friendly voice assistant specialized in interacting with Supabase databases.",
        tools=livekit_tools,
    )

    session = AgentSession(
        vad=silero.VAD.load(min_silence_duration=0.1),
        stt=assemblyai.STT(word_boost=["Supabase"]),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(voice="ash"),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Greet the user and offer to help them with their data in Supabase")

    @ctx._on_shutdown
    async def on_shutdown(ctx: JobContext):
        await server.__aexit__(None, None, None)
        print("Shutting down MCP server")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Understanding the Enhanced Code

Let's break down what we've added:

#### 1. MCP Server Initialization

```python
server = MCPServerStdio(
    "npx",
    args=[
        "-y",
        "@supabase/mcp-server-supabase@latest",
        "--access-token",
        os.environ["SUPABASE_ACCESS_TOKEN"],
    ],
)
```

This code initializes Supabase's MCP server using Node.js (via `npx`). We're using the `pydantic_ai` library which provides a Python interface to MCP servers. The `MCPServerStdio` class creates a subprocess that communicates with the MCP server using standard input/output protool.

#### 2. Converting MCP Tools to LiveKit Function Tools

The `build_livekit_tools` function performs the crucial task of bridging between MCP and LiveKit's function calling system. It:

1. Fetches all available tools from the MCP server using `server.list_tools()`
2. For each tool, generates a proxy function that:
    - Accepts parameters that match the tool's schema
    - Calls the tool with provided arguments
    - Returns the result in the format LiveKit expects
3. Adds proper type annotations and documentation to make the tools work seamlessly with LiveKit

This conversion is necessary because while LiveKit natively supports tools through its function calling system (using Python functions decorated with `@function_tool`), it doesn't yet have native MCP support. MCP defines tools using JSON schemas, so we need to fetch these tools from the MCP server and create corresponding LiveKit-compatible function tools to make them available to our agent.

To give an example, here's the `list_tables` tool definition from the MCP server:
```python
 ToolDefinition(name='list_tables', description='Lists all tables in a schema.', parameters_json_schema={'type': 'object', 'properties': {'project_id': {'type': 'string'}, 'schemas': {'type': 'array', 'items': {'type': 'string'}, 'description': 'Optional list of schemas to include. Defaults to all schemas.'}}, 'required': ['project_id'], 'additionalProperties': False, '$schema': 'http://json-schema.org/draft-07/schema#'}, outer_typed_dict_key=None, strict=None)


```

And here's the equivalent LiveKit function tool which will get generated with `build_livekit_tools`:
```python
@function_tool
async def list_tables(
    context: RunContext,
    project_id: str,
    schemas: List[str] = None
):
    """
    Lists all tables in a schema.
    Args:
        project_id (String): 
        schemas (Optional[List[String]]): Optional list of schemas to include. Defaults to all schemas.
    """
    result = await server.call_tool("list_tables", arguments={"project_id": project_id, "schemas": schemas})
    txt = result.content[0].text
    return json.loads(txt)
```


#### 3. Using the Tools in the Agent

Once converted, we pass the tools to our agent:

```python
agent = Agent(
    instructions="You are a friendly voice assistant specialized in interacting with Supabase databases.",
    tools=livekit_tools,
)
```
We launch and keep the MCP server running in the background with `await server.__aenter__()`.

We've also updated the agent's instructions to indicate that it specializes in Supabase database interactions.

#### 4. Cleanup on Shutdown

Finally, we've added a shutdown handler to properly close the MCP server:

```python
@ctx._on_shutdown
async def on_shutdown(ctx: JobContext):
    await server.__aexit__(None, None, None)
    print("Shutting down MCP server")
```

## Testing Our Database Voice Agent

With this enhanced agent ready, you can now have conversations with it about your Supabase projects.
To run the agent in development mode, you can use the following command:
```bash
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
# Mac/Linux
source venv/bin/activate

# Windows
# .\venv\Scripts\activate.bat

# 3. Install your packages inside the venv
pip install livekit-agents livekit-plugins-assemblyai livekit-plugins-openai livekit-plugins-silero python-dotenv "pydantic-ai-slim[openai,mcp]"

# 4. Run your agent
python agent.py dev
```
If all credentials are set up correctly, you should be able to talk to your agent in the LiveKit Agents Playground and interact with your Supabase projects.

For example, imagine you have a project named `grocery-store`, with a table called products, and a table called sales. You can ask the agent questions like:

- "How many users logged in last month in my `grocery-store` project?"
- "Who is the user with most sales?"
- "Create a `top_products` table with the 10 most sold products"
- "What is the average revenue per month?
- "Create a new record for customer John Smith in the sales table, who bought the product with id 123"

The agent will use the appropriate Supabase tools to fulfill these requests and respond conversationally. 

## Conclusion

In this post, we've built a voice agent that can interact with Supabase databases using LiveKit's agent framework and MCP. We've covered the setup, the code, and the testing process.


