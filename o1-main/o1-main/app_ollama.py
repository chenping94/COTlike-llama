import streamlit as st
import json
import time
import requests
from dotenv import load_dotenv
import os
import re
import traceback

# Load environment variables
load_dotenv()

# Get configuration from .env file
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.2')

def check_for_follow_up(raw_content, step_data):
    if "Please let me know" in raw_content:
        return "Continue, Consider ALL" + important_message
    elif isinstance(step_data, dict) and step_data.get('next_action') == 'continue':
        return 'continue' + important_message
    return None

def extract_json_objects(text):
    # This regex pattern matches JSON-like structures without using recursive patterns
    json_pattern = re.compile(r'\{(?:[^{}]|\{[^{}]*\})*\}')
    return json_pattern.findall(text)

def clean_json_string(json_string):
    # Remove any text before the first '{'
    json_string = re.sub(r'^[^{]*', '', json_string)



    
    # Remove any text after the last '}'
    json_string = re.sub(r'[^}]*$', '', json_string)
    # Remove any trailing commas before closing braces or brackets
    json_string = re.sub(r',\s*([\]}])', r'\1', json_string)
    return json_string

def parse_json_safely(json_string):
    cleaned_json = clean_json_string(json_string)
    json_objects = extract_json_objects(cleaned_json)
    
    if json_objects:
        try:
            # Try to parse the last JSON object found
            return json.loads(json_objects[-1])
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON: {str(e)}")
    
    st.error("No valid JSON object found in the response")
    # return None  # hide if cannot find
    st.text("Raw response:")
    st.code(json_string)
    return {"title": "Error", "content": "Failed to parse response", "next_action": "final_answer"}

def make_api_call(messages, max_tokens, is_final_answer=False):
    for attempt in range(3):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.2
                    }
                }
            )
            response.raise_for_status()
            raw_content = response.json()["message"]["content"]
            parsed_data = parse_json_safely(raw_content)
            return parsed_data, raw_content
        except requests.exceptions.RequestException as e:
            st.error(f"API call failed: {str(e)}")
            st.text("Response content:")
            st.code(response.text if hasattr(response, 'text') else "No response text available")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.text("Traceback:")
            st.code(traceback.format_exc())
        
        if attempt == 2:
            error_message = f"Failed to generate {'final answer' if is_final_answer else 'step'} after 3 attempts."
            return {"title": "Error", "content": error_message, "next_action": "final_answer"}
        time.sleep(1)  # Wait for 1 second before retrying

def generate_response(prompt):
    messages = [
        # {"role": "system", "content": SYSTEM_PROMPT + important_message},
        # {"role": "user", "content": prompt },
        {"role": "system", "content": ""},
        {"role": "user", "content": SYSTEM_PROMPT + important_message + "Here is my first query: " + prompt },
        {"role": "assistant", "content": "Understood. I will now think step by step following the instructions, starting with decomposing the problem. I will provide my response in a single, well-formatted JSON object for each step."}
    ]

    steps = []
    step_count = 1
    total_thinking_time = 0

    while True:
        start_time = time.time()
        step_data, raw_content = make_api_call(messages, 300)
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time

        steps.append((f"Step {step_count}: {step_data['title']}", step_data['content'], thinking_time, raw_content))

        messages.append({"role": "assistant", "content": json.dumps(step_data)})

        # Check if a follow-up is needed
        follow_up = check_for_follow_up(raw_content, step_data)
        if follow_up:
            messages.append({"role": "user", "content": follow_up})
            if follow_up.startswith("continue"):
                step_count += 1
            continue  # Skip to the next iteration without incrementing step_count

        if step_data['next_action'] == 'final_answer':
            break

        step_count += 1

        # Yield after each step for Streamlit to update
        yield steps, None  # We're not yielding the total time until the end

    # Generate final answer
    messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above. Remember to respond with a single, well-formatted JSON object."})

    start_time = time.time()
    final_data, raw_content = make_api_call(messages, 200, is_final_answer=True)
    end_time = time.time()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time

    steps.append(("Final Answer", final_data['content'], thinking_time, raw_content))

    yield steps, total_thinking_time

def main():
    st.set_page_config(page_title="ol1 prototype - Ollama version", page_icon="🧠", layout="wide")

    st.title("ol1: Using Ollama to create o1-like reasoning chains")

    st.markdown("""
    This is an early prototype of using prompting to create o1-like reasoning chains to improve output accuracy. It is not perfect and accuracy has yet to be formally evaluated. It is powered by Ollama so that the reasoning step is local!

    Forked from [bklieger-groq](https://github.com/bklieger-groq)
    Open source [repository here](https://github.com/win4r/o1)
    """)

    st.markdown(f"**Current Configuration:**")
    st.markdown(f"- Ollama URL: `{OLLAMA_URL}`")
    st.markdown(f"- Ollama Model: `{OLLAMA_MODEL}`")

    # Text input for user query
    user_query = st.text_input("Enter your query:", placeholder="e.g., How many 'R's are in the word strawberry?")

    if user_query:
        st.write("Generating response...")

        # Create empty elements to hold the generated text and total time
        response_container = st.empty()
        time_container = st.empty()

        # Generate and display the response
        for steps, total_thinking_time in generate_response(user_query):
            with response_container.container():
                for i, (title, content, thinking_time, raw_content) in enumerate(steps):
                    if title.startswith("Final Answer"):
                        st.markdown(f"### {title}")
                        st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                    else:
                        with st.expander(title, expanded=True):
                            st.markdown(content.replace('\n', '<br>'), unsafe_allow_html=True)
                            st.markdown("**Raw Output:**")
                            st.code(raw_content, language="json")

                            # Check if a follow-up was sent
                            # follow_up = check_for_follow_up(raw_content, json.loads(raw_content))
                            parsed_data = parse_json_safely(raw_content)
                            follow_up = check_for_follow_up(raw_content, parsed_data)
                            if follow_up:
                                if follow_up.startswith("continue"):
                                    st.markdown(f"*Automatic 'continue' prompt sent*")
                                else:
                                    st.markdown(f"*Follow-up prompt sent: '{follow_up}'*")

                    st.markdown(f"*Thinking time: {thinking_time:.2f} seconds*")

            # Only show total time when it's available at the end
            if total_thinking_time is not None:
                time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

SYSTEM_PROMPT = """You are an expert AI assistant with advanced reasoning capabilities. Your task is to provide detailed, step-by-step explanations of your thought process. For each step:

1. Provide a clear, concise title describing the current reasoning phase.
2. Elaborate on your thought process in the content section.
3. Decide whether to continue reasoning or provide a final answer.

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Key Instructions:
- Employ at least 5 distinct reasoning steps such as Edge Case Consideration, Precision Consideration, Alternative Hypothesis or Approach Evaluation and Elimination, etc.
- Acknowledge your limitations as an AI and explicitly state what you can and cannot do.
- Actively explore and evaluate alternative answers or approaches.
- Critically assess your own reasoning; identify potential flaws or biases.
- When re-examining, employ a fundamentally different approach or perspective.
- Utilize at least 3 diverse methods to derive or verify your answer.
- Incorporate relevant domain knowledge and best practices in your reasoning.
- Quantify certainty levels for each step and the final conclusion when applicable.
- Consider potential edge cases or exceptions to your reasoning.
- Provide clear justifications for eliminating alternative hypotheses.

"""
important_message=""" 
IMPORTANT: Respond STRICTLY with a single, well-formatted JSON object for each step. Do not include any text outside the JSON object. Think STEP by STEP. 

Response Format:
Use JSON with keys: 'title', 'content', 'next_action' (values: 'continue' or 'final_answer')

Example of a valid JSON response:
{"title": "Initial Problem Analysis", "content": "To approach this problem effectively, I'll first break down the given information into key components. This involves identifying...[detailed explanation]... By structuring the problem this way, we can systematically address each aspect.", "next_action": "continue"}

"""

if __name__ == "__main__":
    main()