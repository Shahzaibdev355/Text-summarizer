import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pdfplumber
import os

# Define the device for inference (GPU or CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TARGET_LENGTH = 800  # Adjust for a more detailed summary

# Load the fine-tuned model and tokenizer from Hugging Face model hub
fine_tuned_output_dir = "ShahzaibDev/flant5-finetuned-summarizer"
fine_tuned_model = AutoModelForSeq2SeqLM.from_pretrained(fine_tuned_output_dir).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(fine_tuned_output_dir)

# Load the base FLAN-T5 model
base_model_name = "google/flan-t5-base"
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(DEVICE)

# Define the prompt for summarization
def get_prompt(doc):
    """Format prompts for text summarization using FLAN-T5 models."""
    prompt = "Summarize the following document:\n\n"
    prompt += f"{doc}"
    prompt += "\n\n Summary:"
    return prompt

# Generate response (summary) from the model
def get_response(prompt, model, tokenizer):
    """Generate a text summary from the prompt."""
    # Tokenize the prompt
    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding='max_length',
        truncation=True,
        max_length=1024  # Adjust for large inputs
    )

    # Move the inputs to the same device as the model (GPU or CPU)
    model_inputs = encoded_input.to(DEVICE)

    # Generate the response
    generated_ids = model.generate(
        **model_inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=6,  # Increase the number of beams for better diversity
        early_stopping=True,
        no_repeat_ngram_size=3,  # Prevent repetition
        temperature=0.7,  # Increase randomness
        top_k=50  # Control the randomness of the output
    )

    # Decode the response back to text
    decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded_output

# Function to extract text from a PDF file using pdfplumber
def extract_text_from_pdf(pdf_file_path):
    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"The file at {pdf_file_path} does not exist.")
    
    with pdfplumber.open(pdf_file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to trim text to a reasonable length
def trim_text(text, max_words=1400):
    words = text.split()
    return " ".join(words[:max_words])

# Streamlit UI
def main():
    st.title("Text Summarizer with FLAN-T5")

    # Option to input text directly or upload a PDF
    option = st.selectbox("Choose Input Type", ["Enter Text", "Upload PDF"])

    if option == "Enter Text":
        # Text input for summarization
        input_text = st.text_area("Enter the text you want to summarize:", height=200)

        if st.button("Summarize"):
            if input_text:
                prompt = get_prompt(input_text)
                summary = get_response(prompt, fine_tuned_model, tokenizer)
                st.subheader("Generated Summary:")
                st.write(summary)
            else:
                st.error("Please enter some text to summarize.")

    elif option == "Upload PDF":
        # Upload PDF file
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file is not None:
            with open("temp_uploaded_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text from the uploaded PDF
            text = extract_text_from_pdf("temp_uploaded_pdf.pdf")

            # Trim text if necessary (e.g., to 800 words)
            trimmed_text = trim_text(text, max_words=800)

            prompt = get_prompt(trimmed_text)
            summary = get_response(prompt, fine_tuned_model, tokenizer)

            st.subheader("Generated Summary:")
            st.write(summary)
        else:
            st.error("Please upload a PDF file.")

if __name__ == "__main__":
    main()
