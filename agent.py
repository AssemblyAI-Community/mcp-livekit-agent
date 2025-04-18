from __future__ import annotations

import json
import os
from typing import Any, List, Callable
import inspect

from dotenv import load_dotenv
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
            proxy.__doc__ = tool_def.description or ""
            return function_tool(proxy)

        tools.append(make_proxy())  # factory runs *now*, variables frozen

    return tools


# --------------------------------------------------------------------------- #
# LiveKit worker entrypoint                                                   #
# --------------------------------------------------------------------------- #
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
        stt=assemblyai.STT(word_boost = ["Supabase"]),
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
