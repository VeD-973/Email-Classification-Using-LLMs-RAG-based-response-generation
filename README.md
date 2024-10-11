# Technical Assessment - Email Classification and Response System
## Project Overview

This project involves the development of an **AI-driven system** designed to categorize and respond to university emails. It leverages **fine-tuned language models (LLMs)** to classify incoming emails into three main categories:

- Student Queries
- Academic Collaboration Requests
- Corporate Inquiries

Depending on the classification, the system either generates an automatic response using **Retrieval-Augmented Generation (RAG)** or escalates the email to the Head of Department (HOD) for further review and manual response.


## Key Features

1. **Email Classification using Fine-tuned LLMs (OPT-350M)**:
    - Emails are classified into the three predefined categories using a fine-tuned version of Facebook's OPT model.
  
2. **Stacked LSTM Neural Network for Classification**:
    - An additional **Stacked LSTM model** is trained to classify emails, which allows for accurate categorization based on email body content.
    #### Why Stacked LSTMs + Dense Layers for Email Classification?

    **a. Stacked LSTMs for Sequential Data**:
    - **Emails are sequences**: LSTMs handle the sequential nature of emails, capturing word dependencies and context.
    - **Long-range dependencies**: Stacking LSTM layers helps learn both low-level (word) and high-level (sentence) patterns, improving understanding.
    
    **b. Dense Layers for Classification**:
    - **Feature abstraction**: After LSTMs, dense layers process the learned features and map them to categories.
    - **Non-linearity**: Dense layers help form complex decision boundaries for better classification accuracy.
    
    **c. Handles Email Complexity**:
    - **Varied content**: LSTMs adapt well to different email lengths and tones.
    - **Context understanding**: LSTMs learn relationships across the text, while dense layers ensure accurate predictions.


3. **Retrieval-Augmented Generation (RAG)**:
    - **RAG** is used with **LLama-3.2-1B** for generating automated responses to student and academic inquiries.
  
4. **Document Search with FAISS**:
    - A **FAISS-based** similarity search helps in retrieving relevant information from large documents, enabling the system to provide accurate and relevant automated responses.
    - The automated responses are divided into two types:
          - Actual response from the RAG using LLMs for student inquiries and academic enquiries if they are present in the database of the university.
          - Hardcoded response for Corporate level Inquiries. 

5. **Interactive Gradio UI**:
    - A **Gradio-based interface** is provided for users to input email queries, classify them, and generate automated responses.

---

## Installation

### Prerequisites

Ensure that you have Python installed on your machine. You can install the required dependencies by running the following command:

```bash
pip install datasets sentence_transformers PyMuPDF PDFReader pdfplumber faiss-cpu --no-cache langchain pypdf langchain-community streamlit huggingface_hub gradio -U
