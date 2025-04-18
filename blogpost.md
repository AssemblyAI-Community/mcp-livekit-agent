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
2. **Language Understanding (LLM):** The transcribed text is analyzed by an AI language model (like OpenAI's GPT-4) to understand the user's intent and determine how to respond.
3. **Text-to-Speech (TTS):** Finally, the AI's response is converted back into speech using a TTS service, giving our agent a natural voice.

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


If you run the first code piece as is, you'll have a fully functional voice agent capable of natural conversation. While it can already engage in dialogue, it doesn't yet perform specific actions or tasks. Let's address that next by adding function calling capabilities.