AI-Powered Application
This repository contains a simple web application that uses a Python backend and an HTML frontend. It leverages Ollama to access powerful language models for various tasks.

🚀 Prerequisites
Before you begin, you must have Ollama installed on your system. You can find the official installation guide on the Ollama website.

Once Ollama is installed, pull the required language models by running these commands in your terminal:

Bash

ollama pull nomic-embed-text
ollama pull llama3.2:3b
If you prefer to use a different model, remember to update the model name in the app.py file.

🛠️ Setup & Running the Application
Follow these steps to get the application up and running.

Step 1: Install Python Dependencies
First, navigate to the project directory in your terminal and install the required Python libraries using pip:

Bash

pip install -r requirements.txt
Step 2: Start the Backend Server
Next, run the Python script to start the local server.

Bash

python app.py
You'll see a message in your terminal indicating that the server is running, typically on a port like 8000 or 5000.

Step 3: Access the Frontend
With the backend running, open the index.html file in your web browser to access the application.

file:///path/to/your/project/index.html
✨ Usage
The application uses the nomic-embed-text and llama3.2 models from Ollama to process requests from the frontend, enabling text generation and other functionalities. Simply interact with the interface in your browser to start using the application.



## ☁️ Running on Google Colab

You can also run this application in a Google Colab environment. This is particularly useful if you don't want to install dependencies locally.

### Step 1: Access the Colab Notebook

1.  Open the Colab notebook by clicking this link: [https://colab.research.google.com/drive/1MJEEPeeh_GsvQJNd4BGHPbvyms06iHoU?usp=sharing](https://colab.research.google.com/drive/1MJEEPeeh_GsvQJNd4BGHPbvyms06iHoU?usp=sharing)
2.  Once the notebook is open, go to `File` > `Save a copy in Drive`. This creates a personal copy of the notebook in your Google Drive, ensuring you have full control over it.

### Step 2: Run the Notebook Cells

1.  In your new copy of the notebook, navigate to `Runtime` > `Run all`.
2.  Follow the instructions within the notebook. This will set up the environment, install the necessary libraries, and start the backend server.
3.  The notebook will use `ngrok` to create a public URL for your server.

### Step 3: Get the ngrok URL

After the notebook has finished running, it will display a public URL (a `ngrok` link). Copy this link.

### Step 4: Update and Open `index.html`

1.  Download the `index.html` file from this repository to your local machine.
2.  Open the downloaded `index.html` file in a text editor.
3.  Find the `fetch` part of the code and replace the `localhost` URL with the `ngrok` link you copied from the Colab notebook. For example, change `http://localhost:8000` to `https://your-ngrok-url.ngrok-free.app`.
4.  Save the `index.html` file.
5.  Open the modified `index.html` file in your web browser to start interacting with the chatbot.
