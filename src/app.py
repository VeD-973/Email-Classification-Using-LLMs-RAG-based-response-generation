import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pdfplumber
from langchain.chains import LLMChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,TFAutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
import torch
warnings.filterwarnings('ignore')

"""Connecting Hugging face"""

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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" # Please put your own HuggingFace Token

"""Transfering the Dataset and Training the model
6OuzcuVOfCzTA/wOYmTPXh2onygAAAABJRU5ErkJggg==)

Encodings:


1.  0->  Student Enquiries
2.  1->  Academic Collaboration Enquiries
3.  2->  Corporate Enquiries
"""

# Sample DataFrame
dataset = pd.read_csv('dataset/smartSense_TA_UniEmailDataset.csv')
dataset = pd.DataFrame(dataset)
df = dataset[['Email_Body', 'Category']]

# Step 1: Encode the categories
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Category'])

# Step 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['Email_Body'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Step 3: Tokenization and Padding
max_words = 1000  # Vocabulary size
max_len = 50  # Maximum length of sequences

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len, padding='post')

# Step 4: Build the LSTM Model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))  # Embedding layer
model.add(LSTM(128, return_sequences=True))  # First LSTM layer with return_sequences=True
model.add(Dropout(0.5))  # Dropout layer for regularization
model.add(LSTM(128))  # Second LSTM layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(128, activation='relu'))  # New Dense fully connected layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer with softmax

# Compile the Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_padded, y_train, epochs=20, batch_size=4, validation_data=(X_test_padded, y_test))

# Plot Training History
plt.figure(figsize=(14, 5))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

"""As we can see that from the plots, the model is just in the good fit region."""

# Function to predict category of a new sentence
def predict_category(text):
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Predict the category
    prediction = model.predict(padded_sequence)
    predicted_label_index = np.argmax(prediction, axis=1)[0]  # Get the index of the highest probability
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]  # Decode to category

    return predicted_label

"""## **Using RAG (Retreival Augmentation Generation) With LLMs (LLama-3.2-1b) to generate automated response**

Extracting Text from pdf
"""

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()
    return full_text
pdf_path = '/content/drive/MyDrive/SmartSense_TA/CV.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Convert extracted text to LangChain Document
document = Document(page_content=pdf_text)

# Split the document into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents([document])

# Use a larger Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Creating a vector database using FAISS for doing similarity search
db = FAISS.from_documents(chunks, embeddings)

"""Using LLMs with RAG (With proper prompting instructions)"""

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-3.2-1B",
    model_kwargs={
        "temperature": 0.2,  # Adjust for more randomness (0.2 for deterministic)
        "max_length": 180,   # Adjust max length as needed
        "top_k": 5,         # Limit sampling to the top 50 tokens
        "top_p": 0.9,        # Use nucleus sampling with a cumulative probability of 0.9
        "repetition_penalty": 1.2  # Discourage repetition in outputs
    }
)
chain = load_qa_chain(llm, chain_type="stuff")

"""### **Using Gradio (Common UI used for ML related applications)**

After running the below code, open the link in a new tab and proceed with your email queries.

**Note: The initial run if done using CPUs can take upto 2 minutes for the automated response. If GPUs are used this time come down to around half a minute.**
"""

import gradio as gr
# Gradio interface function
def query_system(query):
    predicted_category = predict_category(query)
    if(predicted_category == 'Student Inquiries' or predicted_category == 'Academic Collaboration Inquiries' ):
      docs = db.similarity_search(query)
      ans = chain.run(input_documents=docs, question=query)
      top_answer = ans.split("Helpful Answer:")[1].strip().split('\n')[0] + '\n' + "This response is autogenerated by AI. Please verify information independently before taking any action." # Gets the first answer after "Helpful Answer:"
      return predicted_category, top_answer
    else:
      top_answer = "This is a sensitive email and it will be directly sent to the HOD for personalized manual response."
      return predicted_category, top_answer


# Create the Gradio interface

iface = gr.Interface(
    fn=query_system,
    inputs=gr.Textbox(label="Enter Your Email Query"),  # Custom label for the input field
    outputs=[
        gr.Textbox(label="Predicted Category"),  # Custom label for the predicted category output
        gr.Textbox(label="Automated Reply")  # Custom label for the helpful answer output
    ],
    title="Email Classification and Response System",
    description="Enter an email query to classify and receive a helpful answer."
)

# Launch the interface
iface.launch()

