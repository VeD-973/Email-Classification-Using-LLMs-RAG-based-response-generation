# SmartSense - Email Classification and Response System

## Overview

This project implements an **AI-based system** to categorize and respond to university emails. The system uses **fine-tuned language models (LLMs)** for email classification into three categories:

- Student Inquiries
- Academic Collaboration Inquiries
- Corporate Inquiries

Based on the category, it provides either an automated response (using **Retrieval-Augmented Generation (RAG)**) or escalates the email to the Head of Department (HOD) for manual action.

---

## Key Features

1. **Email Classification using Fine-tuned LLMs (OPT-350M)**:
    - Emails are classified into the three predefined categories using a fine-tuned version of Facebook's OPT model.
  
2. **Stacked LSTM Neural Network for Classification**:
    - An additional **Stacked LSTM model** is trained to classify emails, which allows for accurate categorization based on email body content.

3. **Retrieval-Augmented Generation (RAG)**:
    - **RAG** is used with **LLama-3.2-1B** for generating automated responses to student and academic inquiries.
  
4. **Document Search with FAISS**:
    - A **FAISS-based** similarity search helps in retrieving relevant information from large documents, enabling the system to provide accurate and relevant automated responses.

5. **Interactive Gradio UI**:
    - A **Gradio-based interface** is provided for users to input email queries, classify them, and generate automated responses.

---

## Installation

### Prerequisites

Ensure that you have Python installed on your machine. You can install the required dependencies by running the following command:

```bash
pip install datasets sentence_transformers PyMuPDF PDFReader pdfplumber faiss-cpu --no-cache langchain pypdf langchain-community streamlit huggingface_hub gradio -U
