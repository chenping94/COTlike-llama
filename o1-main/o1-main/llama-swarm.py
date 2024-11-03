# from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm, Agent
import json, sys

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key='ollama'
)

model = 'llama3.2'
modelA = 'qwen2.5-coder:7b'
modelB = 'hf.co/bartowski/Llama-3.1-Nemotron-70B-Instruct-HF-GGUF:IQ1_M'

temperature = 1
top_p = 1
max_tokens = 496

client = Swarm(client=ollama_client)


def llmm (messages):
    completion = ollama_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False
    )
    return completion.choices[0].message.content

def llm (prompt):
    messages = [{"role":"user","content":prompt}]
    completion = ollama_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=False
    )
    return completion.choices[0].message.content

# reply = llm("Hello World")
# print(reply)

agent = Agent(
    name="SuperIntelligence",
    instructions="You are a helpful SuperIntelligence",
    model=model,
    tool_choice="auto"
)

messages = [{"role":"user","content":"hello! Tell me your biggest 5 traits!"}]
# response = client.run(agent=agent, messages=messages)

# print(response.messages[-1]["content"])

def Next():
    return agentB

agentA = Agent(
    name="Planner",
    instructions="You are a helpful Planner.",
    model=model,
    functions = [Next],
    tool_choice="auto"
)

agentB = Agent(
    name="Thinker",
    instructions="You are an extraordinary Thinker. Verify the plans and elaborate on the specific reasoning steps",
    model=modelA,
    tool_choice="auto"
)

response = client.run(
    agent = agentA,
    messages = [{"role":"user","content":"Plan me a honeymoon trip"}]
)

print(response.messages[-1]["content"])