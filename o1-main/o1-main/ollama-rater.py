import os
import json
import requests
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URL = os.getenv('MONGO_URL', 'mongodb://localhost:27017/')
DB_NAME = "COTlike-llama"
COLLECTION_NAME = "steps"

# Ollama configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')

def get_mongo_client():
    client = MongoClient(MONGO_URL)
    return client

def get_database(client, db_name):
    return client[db_name]

def get_steps_data(db):
    collection = db[COLLECTION_NAME]
    steps_data = collection.find_one(sort=[('_id', -1)])  # Get the latest document
    return steps_data

def make_api_call(messages):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": 500,
                "temperature": 0.2
            }
        }
    )
    response.raise_for_status()
    return response.json()

def main():
    client = get_mongo_client()
    db = get_database(client, DB_NAME)
    steps_data = get_steps_data(db)

    if not steps_data:
        print("No steps data found in MongoDB.")
        return

    # Prepare the prompt
    steps_json = json.dumps(steps_data['steps'])
    prompt = f"{RATER_PROMPT}\n\nSteps Data:\n{steps_json}"

    messages = [
        {"role": "system", "content": "You are an expert critic and response reflector."},
        {"role": "user", "content": prompt}
    ]

    # Make API call to the model
    response = make_api_call(messages)
    feedback = response["message"]["content"]

    # Store the feedback in MongoDB
    collection = db["feedback"]
    collection.insert_one({"feedback": feedback})

    print("Feedback stored successfully.")

# Rater prompt
RATER_PROMPT = '''
As an expert critic and LLM reflector, your task is to analyze the step-by-step response of an expert in specified domain towards a query, identifying specific areas where the response may lack clarity, depth, or relevance, and providing constructive feedback.

Focus on providing detailed and constructive feedback, highlighting shortcomings and offering actionable suggestions for improvement. Maintain a supportive tone that encourages growth and development.

# Assessment Criteria

- **Logical**: Does the reasoning steps logical? Do the values or evidences used are in accurate and logical manner? Have it considered all of Edge Case Consideration, Precision Consideration, Alternative Hypothesis or Approach Evaluation and Elimination?
- **Clarity**: Does the response clearly convey information and ideas? Are there any ambiguous or confusing sections?
- **Depth**: To what extent does the response explore the topic? Are complex ideas and edge cases fully developed and well-considered?
- **Relevance**: How directly does the response address the prompt or topic? Are there any areas where the response deviates without purpose?
- **Coherence**: Are the ideas and arguments presented in a logically consistent manner? Does the flow of information make sense?
- **Accuracy**: Are the facts and data presented correct and up-to-date? Are sources of information trustworthy? How many confidence level would you rate?
- **Usefulness**: Does the response offer unique insights or express ideas in an useful way, is the solution offered feasible to address the issues?

# Output Format

Produce a well-formatted JSON for each assessment criterion listed, offering detailed feedback. Recap the key strengths and areas for improvement.
Use JSON with keys: 'title' (values: Clarity,Depth,Relevance,Coherence,Accuracy,Usefulness), 'comment', 'rating' (values: ranges from 0 to 1 in 2 decimal place, e.g. 0.15, 0.25, 0.30, ...)

# Examples

- **Input**: "Query: 'What is the fox in this picture doing?' Expert response: 'The quick brown fox jumps over the lazy dog. It is known that foxes are part of the Canidae family.'"

{"title": "Logical", "comment": "During Step 1 Problem Decomposition ought to consider more than 3 distinct cases, and the values used are questionable. Provided on 19XX, the values is actually XX...", "rating": "0.55"}
{"title": "Clarity", "comment": "The initial sentence is simple and clear, though simplistic. The subsequent sentence introduces a fact but lacks context linking it to the previous statement.", "rating": "0.65"}
{"title": "Depth", "comment": "The response provides minimal exploration of foxes or the significance of the phrase introduced.", "rating": "0.50"}
{"title": "Relevance", "comment": "While factual, the information on the Canidae family seems tangential to the core topic presented by the phrase.", "rating": "0.65"}
{"title": "Coherence", "comment": "The transition between sentences could be smoother with a connective rationale.", "rating": "0.75"}
{"title": "Accuracy", "comment": "The statement about foxes is factually accurate.", "rating": "0.95"}
{"title": "Usefulness", "comment": "The response lacks usefulness, largely restating known information without unique insight (out of the box) that could really solve the issues.", "rating": "0.45"}

**Recap**: The response is clear and accurate but fails to provide engaging or deeply explored content. Consider expanding the context and integrating creative links between ideas.

# Notes

- Encourage improvements by suggesting specific changes, like adding examples or contextual explanations.
- Maintain a positive and encouraging tone throughout the feedback.
- Address both strengths and weaknesses equally to provide balanced feedback.

'''

if __name__ == "__main__":
    main()