# backend.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
import uuid
import json
from datetime import datetime
import os

# Initialize the FastAPI application
app = FastAPI(title="Kisan AI Advisor Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
chroma_client_schemes = chromadb.PersistentClient(path="./vector_dbs/chroma_db_schemes")
chroma_client_pp_qna = chromadb.PersistentClient(path="./vector_dbs/chroma_db_pp_qna")
chroma_client_pp = chromadb.PersistentClient(path="./vector_dbs/chroma_db_plant_protection")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
#import pdb; pdb.set_trace()

# Initialize collections
schemes_collection = chroma_client_schemes.get_or_create_collection("agri_schemes")
protection_collection = chroma_client_pp.get_or_create_collection("plant_protection")
pest_collection = chroma_client_pp_qna.get_or_create_collection("crop_qna_knowledge_base")
chat_history_collection = chroma_client.get_or_create_collection("chat_history")
# Initialize Ollama embeddings
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
except:
    print("Warning: Could not initialize Ollama embeddings. Using mock embeddings.")
    embeddings = None

# In-memory chat sessions storage
chat_sessions = {}



# --- Tool Definitions with @tool decorator ---

@tool
def get_weather(input: str = "") -> str:
    """Get current weather information for farming activities."""
    print('weather tool triggered')
    print(input)
    return """## 🌤️ Current Weather Report

**Temperature:** 28°C  
**Conditions:** Partly Cloudy with a chance of thunderstorms  
**Humidity:** 75%  
**Wind Speed:** 10 km/h  

### 🚜 Farming Recommendations:
- **Good conditions** for most farming activities
- **Consider covering** sensitive crops if thunderstorms develop
- **Ideal time** for irrigation due to high humidity
- **Monitor weather** updates for storm warnings"""

@tool
def get_mandi_prices(input: str = "") -> str:
    """Get current mandi (market) prices for agricultural commodities."""
    
    print('mandi tool triggered')
    return """## 💰 Live Mandi Prices

| Commodity | Location | Price (₹/quintal) | Trend |
|-----------|----------|------------------|--------|
| **Wheat** | Indore | ₹2,500 | ↗️ +2.5% |
| **Rice** | Amritsar | ₹8,200 | ↘️ -1.2% |
| **Potato** | Agra | ₹1,850 | ↗️ +5.8% |
| **Onion** | Nashik | ₹3,200 | → Stable |
| **Tomato** | Kolar | ₹4,500 | ↗️ +12.3% |

### 📈 Market Insights:
- **Potato** and **Tomato** showing strong upward trends
- **Rice** prices slightly down due to good harvest
- **Wheat** remains steady with marginal increase
- *Last updated: Today at 2:00 PM*"""

@tool
def search_schemes(query: str) -> str:
    """Search for government schemes and policies , yojana """
    try:
        print('search schemes triggered') 
        if embeddings:
            query_embedding = embeddings.embed_query(query)
            results = schemes_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
        else:
            # Fallback to simple text search
            results = schemes_collection.query(
                query_texts=[query],
                n_results=2
            )
           
        #import pdb;pdb.set_trace()
        if results['documents'][0]:
            formatted_result = ""
            for i, doc in enumerate(results['documents'][0]):
                formatted_result += "\n{doc}\n\n"
            formatted_result +=str(results['metadatas'])
            #import pdb;pdb.set_trace()
            return formatted_result.strip()
        else:
            return "## ❌ No Information Found\n\nNo specific scheme information found for your query. Please try with different keywords."
    except Exception as e:
        return f"Error searching schemes: {str(e)}"

@tool
def search_protection(query: str) -> str:
    """Search for crop protection and farming best practices information, any information realted to bugs , or crop health , pests and similiar"""
    try:
        if embeddings:
            query_embedding = embeddings.embed_query(query)
            results = protection_collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
        else:
            results = protection_collection.query(
                query_texts=[query],
                n_results=3
            )
        if embeddings:
            query_embedding = embeddings.embed_query(query)
            results_qna = pest_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
        else:
            results_qna = pest_collection.query(
                query_texts=[query],
                n_results=2
            )
        #import pdb;pdb.set_trace()
        if results_qna['documents'][0]:
            formatted_result_qna = "Based on the query from user , find the most similiar question answer pairs and use them in the final answer , Always cite the source if avialbale\n\n"
            for i, doc in enumerate(results_qna['documents'][0]):
                formatted_result_qna += f"\n{doc}\n\n {str(results_qna['metadatas'][0][i])}"
            #import pdb;pdb.set_trace()
        if results['documents'][0]:
            formatted_result = "## Use the dat below as a guide for crop protection and farming best practices neglect if information is not relevant to user question,cite the source if avialbale\n\n"
            for i, doc in enumerate(results['documents'][0]):
                formatted_result += f"\n{doc}\n\n {str(results['metadatas'][0][i])}"
            formatted_result += formatted_result_qna
            print(formatted_result)
            return formatted_result.strip()
        else:
            return "## ❌ No Information Found\n\nNo specific protection information found for your query. Please try with different keywords."
    except Exception as e:
        return f"Error searching protection info: {str(e)}"

@tool
def search_pest_info(query: str) -> str:
    """Search for pest identification and management information."""
    try:
        if embeddings:
            query_embedding = embeddings.embed_query(query)
            results = pest_collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
        else:
            results = pest_collection.query(
                query_texts=[query],
                n_results=5
            )
        
        if results['documents'][0]:
            print(results['documents'][0])
            formatted_result = "## 🐛 Pest Management Information\n\n"
            for i, doc in enumerate(results['documents'][0]):
                formatted_result += f"### {i+1}. Pest Control Method\n{doc}\n\n"
            return formatted_result.strip()
        else:
            return "## ❌ No Information Found\n\nNo specific pest information found for your query. Please try with different keywords."
    except Exception as e:
        return f"Error searching pest info: {str(e)}"

# List of all tools
tools = [get_weather, get_mandi_prices, search_schemes, search_protection, search_pest_info]
tool_node = ToolNode(tools)

# --- Memory Management Functions ---

def save_chat_history(session_id: str, message: BaseMessage):
    """Save chat message to ChromaDB for persistent memory."""
    try:
        message_id = f"{session_id}_{datetime.now().isoformat()}"
        message_data = {
            "session_id": session_id,
            "content": message.content,
            "type": message.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
        
        chat_history_collection.add(
            documents=[message.content],
            metadatas=[message_data],
            ids=[message_id]
        )
    except Exception as e:
        print(f"Error saving chat history: {e}")

def load_chat_history(session_id: str) -> List[BaseMessage]:
    """Load chat history from ChromaDB."""
    try:
        results = chat_history_collection.get(
            where={"session_id": session_id},
            limit=20  # Limit to last 20 messages
        )
        messages = []
        if results['metadatas']:
            # Sort by timestamp
            sorted_results = sorted(
                zip(results['documents'], results['metadatas']),
                key=lambda x: x[1]['timestamp']
            )
            for doc, metadata in sorted_results:
                if metadata['type'] == 'HumanMessage':
                    messages.append(HumanMessage(content=doc))
                elif metadata['type'] == 'AIMessage':
                    messages.append(AIMessage(content=doc))
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

# --- Define the LangGraph State ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    session_id: str

# --- Node Functions ---

def tool_router(state: AgentState) -> Dict[str, Any]:
    """
    Use LLM to decide which tool(s) to call, similar to a supervisor agent.
    """
    last_message = state["messages"][-1].content

    tool_descriptions = [
        "get_weather: Get current weather information for farming activities.",
        "get_mandi_prices: Get current mandi (market) prices for agricultural commodities.look for mandi , bhaav , rate keyword",
        "search_schemes: Search for government schemes and policies related to agriculture.",
        "search_protection: Search for crop protection and farming best practices information, any information realted to bugs , or crop health , pests and similiar",
        #"search_pest_info: Search for pest identification and management information."
    ]
    tool_list = "\n".join(f"- {desc}" for desc in tool_descriptions)

    prompt = f"""
You are Kisan AI, a friendly and expert agricultural assistant for Indian farmers.
Given the user's message, decide which of the following tools (if any) should be called. 
List the tool names that are most relevant, or say "none" if none apply.

Available tools:
{tool_list}

User message: {last_message}

Respond with a comma-separated list of tool names, or "none".
"""

    ollama_model = Ollama(model="llama3.2:3b")
    response = ollama_model.invoke(prompt).strip().lower()

    tool_calls = []
    if response != "none":
        for tool_name in [t.strip() for t in response.split(",")]:
            if tool_name in ["get_weather", "get_mandi_prices"]:
                tool_calls.append({
                    "id": f"call_{tool_name}_{len(state['messages'])}",
                    "name": tool_name,
                    "args": {"input": ""}
                })
            elif tool_name in ["search_schemes", "search_protection", "search_pest_info"]:
                tool_calls.append({
                    "id": f"call_{tool_name}_{len(state['messages'])}",
                    "name": tool_name,
                    "args": {"query": last_message}
                })
    # Always return tool_calls (empty if none)
    ai_message = AIMessage(content="Let me help you with that information.", tool_calls=tool_calls)
    return {"messages": state["messages"] + [ai_message]}

def tool_and_summarize_node(state: AgentState) -> Dict[str, Any]:
    """
    Executes the tool(s) and then lets the LLM summarize/format the results.
    """
    last_ai_message = state["messages"][-1]
    tool_calls = getattr(last_ai_message, "tool_calls", [])
    tool_outputs = []
    print(tool_calls)
    
    for call in tool_calls:
        tool_name = call["name"]
        args = call.get("args", {})
        
        # Call the tool by name
        tool_func = next((t for t in tools if t.name == tool_name), None)
        if tool_func:
            try:
                # Handle different tool signatures properly
                if tool_name in ["get_weather", "get_mandi_prices"]:
                    # These tools expect a single 'input' parameter
                    input_value = args.get("input", "")
                    result = tool_func(input_value)
                elif tool_name in ["search_schemes", "search_protection", "search_pest_info"]:
                    # These tools expect a single 'query' parameter
                    query_value = args.get("query", "")
                    result = tool_func(query_value)
                else:
                    # Fallback: try to call with no arguments
                    result = tool_func()
                    
            except Exception as e:
                result = f"Error running tool {tool_name}: {str(e)}"
            tool_outputs.append({"tool": tool_name, "output": result})
        else:
            tool_outputs.append({"tool": tool_name, "output": f"Tool {tool_name} not found."})

    # Compose a prompt for the LLM to summarize/format the tool outputs
    user_message = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
    tool_outputs_str = "\n\n".join(
        f"Tool: {o['tool']}\nOutput:\n{o['output']}" for o in tool_outputs
    )

    prompt = f"""You are Kisan AI, a friendly and expert agricultural assistant for Indian farmers.

Here is the user's question:
{user_message}

Here are the results from the tools you called:
{tool_outputs_str}

**Instructions for Kisan AI:**

- **Greeting:** Start with a warm and friendly greeting that acknowledges the user's query.
- **Summary & Relevance:** Summarize the information from the tool outputs in a clear, helpful, and regionally relevant way. Tailor the language and examples to the context of Indian farming.
- **Formatting:** Use markdown formatting to make the information easy to read and understand.
  - Use headings (##) for different sections (e.g., "Weather Update," "Pest Control," "Market Prices").
  - Use bullet points (or lists) and tables where appropriate to present data clearly.
  - Incorporate relevant emojis (e.g., ☀️, 🌧️, 🌱, 🚜) to make the response more engaging.
  - Use **bold** text to highlight important keywords or numbers.
  - Ensure newlines are used to separate paragraphs and sections for better readability.
- **Coherence:** If multiple tools were used, combine their outputs into a single, coherent, and logical answer. Avoid redundancy.
- **Clarity & Tone:** Maintain a clear, simple, and encouraging tone. Use straightforward language and avoid technical jargon.
- **Closing:** End with a helpful and encouraging closing statement.

Respond as Kisan AI, following all the above instructions.
"""

    ollama_model = Ollama(model="llama3.2:3b")
    response = ollama_model.invoke(prompt)
    ai_message = AIMessage(content=response)
    return {"messages": state["messages"] + [ai_message]}
# --- Build the LangGraph ---

workflow = StateGraph(AgentState)

workflow.add_node("router", tool_router)
workflow.add_node("tools_and_summarize", tool_and_summarize_node)

workflow.set_entry_point("router")

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools_and_summarize"
    return END

workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "tools_and_summarize": "tools_and_summarize",
        END: END
    }
)

workflow.add_edge("tools_and_summarize", END)

app_graph = workflow.compile()

# Visualize as PNG
from PIL import Image
from io import BytesIO

png_data = app_graph.get_graph().draw_png()
arch = Image.open(BytesIO(png_data))
arch.save("workflow_akash.png")
#import pdb; pdb.set_trace()
# --- FastAPI Models ---


# --- FastAPI Models ---

class ChatMessage(BaseModel):
    message: str
    session_id: str = None

# --- FastAPI Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """Main chat endpoint with memory support."""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Load existing chat history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = load_chat_history(session_id)
        
        # Add user message
        user_message = HumanMessage(content=request.message)
        chat_sessions[session_id].append(user_message)
        save_chat_history(session_id, user_message)
        
        # Prepare state for the graph
        state = {
            "messages": chat_sessions[session_id],
            "session_id": session_id
        }
        
        # Run through the graph
        result = app_graph.invoke(state)
        
        # Get the final AI response
        ai_response = result["messages"][-1]
        
        # Update session history
        chat_sessions[session_id] = result["messages"]
        save_chat_history(session_id, ai_response)
        
        return {
            "response": ai_response.content,
            "session_id": session_id
        }
    
    except Exception as e:
        error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
        return {
            "response": error_response,
            "session_id": request.session_id or str(uuid.uuid4())
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a specific session."""
    try:
        # Remove from memory
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        # Remove from ChromaDB
        chat_history_collection.delete(where={"session_id": session_id})
        
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear chat history: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)