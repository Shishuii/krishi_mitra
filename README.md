README
This document provides instructions on how to set up and run the project. The application consists of a backend Python server and a frontend HTML interface.

🚀 Prerequisites
Before you can run the application, you need to have Ollama installed on your system. If you don't have it, please follow the official installation guide for your operating system.

Once Ollama is installed, you must pull the required language models by running the following commands in your terminal:

ollama pull nomic-embed-text
ollama pull llama3.2:3b 

Can use any other tool calling enable llm as well.
just cahnge in app.py
🛠️ Setup & Running the Application
Step 1: Install Python Dependencies
Navigate to the project directory in your terminal and install the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

Step 2: Start the Backend
Now, you can run the Python backend script. This will start a local server that the frontend will connect to.

python app.py

You should see output indicating that the server is running, likely on a port like 8000 or 5000.

Step 3: Access the Application
With the backend server running, you can now open the frontend. Simply open the index.html file in your preferred web browser.

file:///path/to/your/project/index.html

✨ Usage
Once the page loads, you can interact with the application. The backend server uses the nomic-embed-text and llama3.1 models to process requests from the frontend, enabling various functionalities such as text generation or embeddings.
