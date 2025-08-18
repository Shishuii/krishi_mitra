# backend.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, Annotated, List
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
chroma_client = chromadb.PersistentClient(path="./chroma_db")
#import pdb; pdb.set_trace()

# Initialize collections
schemes_collection = chroma_client.get_or_create_collection("agri_schemes")
protection_collection = chroma_client.get_or_create_collection("plant_protection")
chat_history_collection = chroma_client.get_or_create_collection("chat_history")

# Initialize Ollama embeddings
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
except:
    print("Warning: Could not initialize Ollama embeddings. Using mock embeddings.")
    embeddings = None

# In-memory chat sessions storage
chat_sessions = {}

# Initialize RAG data in ChromaDB (run once)
def initialize_rag_data():
    """Initialize the vector database with sample data."""
    
    schemes_data = [
        {
            "id": "scheme_1",
            "text": "Pradhan Mantri Fasal Bima Yojana (PMFBY) offers comprehensive crop insurance coverage against natural calamities, pests, and diseases. It provides financial support to farmers in case of crop failure.",
            "metadata": {"category": "insurance", "scheme": "PMFBY"}
        },
        {
            "id": "scheme_2", 
            "text": "Kisan Credit Card (KCC) provides timely and adequate credit support for agricultural and allied activities. It offers flexible repayment options and competitive interest rates.",
            "metadata": {"category": "credit", "scheme": "KCC"}
        },
        {
            "id": "scheme_3",
            "text": "Soil Health Card Scheme promotes soil testing and provides recommendations for appropriate nutrient management. It helps farmers optimize fertilizer use and improve crop productivity.",
            "metadata": {"category": "soil", "scheme": "SHC"}
        }
    ]
    
    protection_data = [
        {
            "id": "protection_1",
            "text": "Integrated Pest Management (IPM) combines biological, cultural, physical, and chemical tools to minimize pesticide use while maintaining crop yields. It's an environmentally sustainable approach.",
            "metadata": {"category": "pest_management", "type": "IPM"}
        },
        {
            "id": "protection_2",
            "text": "Crop rotation helps prevent soil-borne diseases, reduces pest buildup, and improves soil fertility. Different crops have varying nutrient requirements and pest vulnerabilities.",
            "metadata": {"category": "cultivation", "type": "rotation"}
        },
        {
            "id": "protection_3",
            "text": "Biopesticides are eco-friendly alternatives derived from natural materials like animals, plants, bacteria, and minerals. They pose minimal risk to humans and beneficial insects.",
            "metadata": {"category": "pesticides", "type": "biological"}
        }
    ]
    
    # Add data to collections if they're empty
    try:
        if schemes_collection.count() == 0:
            for item in schemes_data:
                if embeddings:
                    embedding = embeddings.embed_query(item["text"])
                    schemes_collection.add(
                        embeddings=[embedding],
                        documents=[item["text"]],
                        metadatas=[item["metadata"]],
                        ids=[item["id"]]
                    )
        
        if protection_collection.count() == 0:
            for item in protection_data:
                if embeddings:
                    embedding = embeddings.embed_query(item["text"])
                    protection_collection.add(
                        embeddings=[embedding],
                        documents=[item["text"]],
                        metadatas=[item["metadata"]],
                        ids=[item["id"]]
                    )
        
        
    except Exception as e:
        print(f"Error initializing RAG data: {e}")

# Initialize data on startup
initialize_rag_data()

# --- Tool Definitions with @tool decorator ---

@tool
def get_weather() -> str:
    """Get current weather information for farming activities."""
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
def get_mandi_prices() -> str:
    """Get current mandi (market) prices for agricultural commodities."""
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
    """Search for government schemes and policies related to agriculture."""
    try:
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
        
        if results['documents'][0]:
            formatted_result = "## 🏛️ Government Schemes Information\n\n"
            for i, doc in enumerate(results['documents'][0]):
                formatted_result += f"### {i+1}. Scheme Details\n{doc}\n\n"
            return formatted_result.strip()
        else:
            return "## ❌ No Information Found\n\nNo specific scheme information found for your query. Please try with different keywords."
    except Exception as e:
        return f"Error searching schemes: {str(e)}"

@tool
def search_protection(query: str) -> str:
    """Search for crop protection farming practices organic methods and sustainable agriculture, 
    Search for pest identification management disease control and insect problems in crops."""
    try:
        
        print(' search protection tool call triggered')
        if embeddings:
            query_embedding = embeddings.embed_query(query)
            results = protection_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
            
        else:
            results = protection_collection.query(
                query_texts=[query],
                n_results=2
            )
        
        if results['documents'][0]:
            formatted_result = "## 🛡️ Crop Protection Information\n\n"
            print(results)
            for i, doc in enumerate(results['documents'][0]):
                formatted_result += f"### {i+1}. Protection Method\n{doc}\n\n"
            return formatted_result.strip()
        else:
            return "## ❌ No Information Found\n\nNo specific protection information found for your query. Please try with different keywords."
    except Exception as e:
        return f"Error searching protection info: {str(e)}"

# List of all tools
tools = [get_weather, get_mandi_prices, search_schemes, search_protection]
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
        results = chat_history_collection.query(
            where={"session_id": session_id},
            n_results=20  # Limit to last 20 messages
        )
        
        messages = []
        if results['metadatas'][0]:
            # Sort by timestamp
            sorted_results = sorted(
                zip(results['documents'][0], results['metadatas'][0]),
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

def tool_router(state: AgentState):
    """Route the user's query to appropriate tools."""
    last_message = state["messages"][-1].content.lower()
    
    tools_to_call = []
    
    # Determine which tools to call based on keywords
    if any(keyword in last_message for keyword in ["weather", "climate", "mausam", "temperature", "rain"]):
        tools_to_call.append("get_weather")
    
    if any(keyword in last_message for keyword in ["mandi", "prices", "bhav", "market", "cost","msp"]):
        tools_to_call.append("get_mandi_prices")
    
    if any(keyword in last_message for keyword in ["scheme", "yojana", "sarkari", "government", "policy"]):
        tools_to_call.append("search_schemes")
    
    if any(keyword in last_message for keyword in ["protection", "suraksha", "fasal", "crop care"]):
        tools_to_call.append("search_protection")
    
    
    
    if tools_to_call:
        # Create tool call messages
        from langchain_core.messages import ToolMessage
        import json
        
        tool_calls = []
        for tool_name in tools_to_call:
            tool_calls.append({
                "id": f"call_{tool_name}_{len(state['messages'])}",
                "name": tool_name,
                "args": {"query": last_message} if tool_name.startswith("search_") else {}
            })
        
        # Add AI message with tool calls
        ai_message = AIMessage(content="Let me help you with that information.", tool_calls=tool_calls)
        return {"messages": [ai_message]}
    else:
        # Use Ollama for general queries
        return call_ollama(state)

def call_ollama(state: AgentState):
    """Call Ollama for general agricultural advice."""
    try:
        ollama_model = Ollama(model="llama3.2:3b")
        
        # Get recent context from chat history
        recent_messages = state["messages"][-5:]  # Last 5 messages for context
        context = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in recent_messages])
        
        prompt = f"""You are an expert agricultural advisor helping Indian farmers. Provide helpful, practical advice in a friendly tone using markdown formatting for better readability.

Use these formatting guidelines:
- Use ## for main headings
- Use ### for sub-headings  
- Use **bold** for important points
- Use bullet points with - for lists
- Use numbered lists when showing steps
- Use tables when comparing information
- Use emojis sparingly for visual appeal
- Answer in hindi language 
Previous conversation context:
{context}

Current question: {state["messages"][-1].content}

Provide a well-formatted response with practical farming advice:"""
        
        response = ollama_model.invoke(prompt)
        ai_message = AIMessage(content=response)
        return {"messages": [ai_message]}
    
    except Exception as e:
        error_message = AIMessage(content=f"I'm having trouble connecting to my knowledge base. Please ensure the AI service is running. Error: {str(e)}")
        return {"messages": [error_message]}

# --- Build the LangGraph ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("router", tool_router)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("router")

# Add conditional edges
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

workflow.add_conditional_edges(
    "router",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", END)

# Compile the graph
app_graph = workflow.compile()

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
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)