from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm
import json

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key='ollama'
)


client = Swarm(client=ollama_client)