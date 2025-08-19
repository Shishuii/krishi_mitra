# AI-Powered Application

This repository contains a simple web application that uses a Python backend and an HTML frontend. It leverages **Ollama** to access powerful language models for various tasks.

-----

## 🚀 Prerequisites

Before you begin, you must have **Ollama** installed on your system. You can find the official installation guide on the [Ollama website](https://ollama.com/download).

Once Ollama is installed, pull the required language models by running these commands in your terminal:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

If you prefer to use a different model, remember to update the model name in the `app.py` file.

-----

## 🛠️ Setup & Running the Application

Follow these steps to get the application up and running.

### Step 1: Install Python Dependencies

First, navigate to the project directory in your terminal and install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Step 2: Start the Backend Server

Next, run the Python script to start the local server.

```bash
python app.py
```

You'll see a message in your terminal indicating that the server is running, typically on a port like `8000` or `5000`.

### Step 3: Access the Frontend

With the backend running, open the `index.html` file in your web browser to access the application.

```
file:///path/to/your/project/index.html
```

-----

## ✨ Usage

The application uses the `nomic-embed-text` and `llama3.2` models from Ollama to process requests from the frontend, enabling text generation and other functionalities. Simply interact with the interface in your browser to start using the application.
