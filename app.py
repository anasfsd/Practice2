# Import necessary libraries
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import os  # To handle environment variables

# Load Hugging Face token from environment variables
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    raise ValueError("Hugging Face token not found. Set the HUGGINGFACE_TOKEN environment variable.")

# Authenticate using the token (in case it's needed for loading a private model)
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Consider using a health-specific model if available

# Determine if a GPU is available
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

# Create the text generation pipeline securely
generator = pipeline("text-generation", model=MODEL_NAME, device=device, use_auth_token=HUGGINGFACE_TOKEN)

# Set up the Streamlit app UI
st.title("Symptom Checker and Health Guidance")
st.write("Enter your symptoms below to receive AI-generated insights. Please note that this is not a substitute for professional medical advice.")

# User input for symptoms
symptom_input = st.text_area("Describe your symptoms here (e.g., 'I have a sore throat and a mild fever.')")

# Generate response on button click
if st.button("Analyze Symptoms"):
    if symptom_input.strip():  # Ensure input is not empty or just whitespace
        # Add a disclaimer prompt to clarify this is not medical advice
        disclaimer_prompt = (
            "You are an AI language model trained to assist with health-related questions. Provide insights "
            "based on common health information but avoid diagnosing. Remind the user that consulting a "
            "medical professional is essential."
        )

        # Combine disclaimer prompt with user input
        input_text = f"{disclaimer_prompt}\n\nUser: {symptom_input}\nAI:"

        # Generate a response
        response = generator(input_text, max_length=150, num_return_sequences=1)[0]['generated_text']

        # Post-process the response to exclude the disclaimer and input prompt
        response_text = response.split("AI:")[-1].strip()

        # Display the AI response
        st.write("### AI's Response")
        st.write(response_text)

        # Disclaimer at the bottom
        st.warning("**Disclaimer:** This AI tool is not a substitute for professional medical advice. "
                   "Please consult a healthcare provider for an accurate diagnosis and treatment.")
    else:
        st.error("Please enter your symptoms for analysis.")
