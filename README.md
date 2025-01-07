# BrailleCart: AI-Powered Grocery Assistance for the Visually Impaired

## Introduction

**BrailleCart** is an AI-powered system designed to assist visually impaired individuals in identifying and learning about grocery items in real-time. Combining advanced AI technologies like YOLOv8n for object detection, OCR for extracting product details, and LLaMA-based conversational text-to-speech, BrailleCart enhances the shopping experience through a simple and accessible interface.


## Dataset
[Dataset download link](https://universe.roboflow.com/new-workspace-wfzw3/grocery-dataset-q9fj2/dataset/5)

## Key Features
- **Real-Time Object Detection**: Utilizes YOLOv8n for precise and quick identification of grocery items.
- **Text Recognition (EasyOCR)**: Extracts product names, prices, and other details from labels.
- **Conversational Feedback**: Uses LLaMA to generate natural language descriptions of detected items.
- **Audio Accessibility**: Provides real-time audio feedback with Google Text-to-Speech (gTTS).
- **User-Friendly Interface**: Streamlit-based UI for seamless interaction.


## Steps to Run the Appliction

- Clone the repository:
   ```bash
   git clone https://github.com/sudip0789/BrailleCart.git
   cd BrailleCart

- Replace the hugging face token with your own HF token

- Start the application:
    ```bash
    streamlit run braillecart_app.py

