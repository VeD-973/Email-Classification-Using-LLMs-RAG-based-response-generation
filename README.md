Email Classification and Automated Response System
Project Overview
This project is designed to fine-tune a pre-trained large language model (LLM) for classifying university emails and generating automated responses. The system classifies incoming emails into predefined categories such as student inquiries, academic collaborations, and corporate queries. Depending on the category, the system either generates a helpful response using a Retrieval-Augmented Generation (RAG) approach or escalates the email for manual review.

The system is built using a combination of several technologies:

Hugging Face Models for classification and text generation.
FAISS for similarity-based document retrieval.
LSTM for email classification.
Gradio for creating an intuitive user interface.
Features
Email Classification: The model classifies emails into 3 categories:
Student Inquiries
Academic Collaboration Inquiries
Corporate Enquiries
RAG-Enabled Responses: For specific categories, the system generates automated responses using LLMs and knowledge retrieval.
Gradio Interface: A user-friendly interface that allows users to input email queries and get a classification and response.
PDF Text Extraction: Text can be extracted from PDF documents for processing and response generation.
Installation
Prerequisites
Python 3.7+
GPU (Optional but recommended for faster response generation)
Required Libraries
You can install the required dependencies by running the following command:
