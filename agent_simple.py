from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import assemblyai, openai

from dotenv import load_dotenv
load_dotenv()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    agent = Agent(
        instructions="You are a friendly voice assistant",
    )

    session = AgentSession(
        stt=assemblyai.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(),
    )

    await session.start(agent=agent, room=ctx.room)
    await session.generate_reply(instructions="Greet the user and ask them about their day")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
