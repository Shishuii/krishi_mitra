# backend.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, Annotated, List, Literal
import operator
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_community.embeddings import OllamaEmbeddings
import chromadb
from chromadb.config import Settings
import uuid
import json
from datetime import datetime
import os
import re

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
                        ids=[item["id"]]                    )
        
        
    except Exception as e:
        print(f"Error initializing RAG data: {e}")

# Initialize data on startup
initialize_rag_data()

# --- Tool Definitions with @tool decorator ---

@tool
def get_weather() -> str:
    """Get current weather information for farming activities and climate conditions."""
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
    """Get current mandi market prices for agricultural commodities and crops."""
    print("Fetching live mandi prices...")
    return """## 💰 Live Mandi Prices

| Commodity | Location | Price (₹/quintal) | Trend |
|-----------|----------|------------------|--------|
| **Wheat** | Indore | ₹2,500 | ↗️ +2.5% |
| **Rice** | Amritsar | ₹8,200 | ↘️ -1.2% |
| **Potato** | Agra | ₹1,850 | ↗️ +5.8% |
| **Onion** | Nashik | ₹3,200 | ↗ Stable |
| **Tomato** | Kolar | ₹4,500 | ↗️ +12.3% |

### 📈 Market Insights:
- **Potato** and **Tomato** showing strong upward trends
- **Rice** prices slightly down due to good harvest
- **Wheat** remains steady with marginal increase
- *Last updated: Today at 2:00 PM*"""

@tool
def search_schemes(query: str) -> str:
    """Search for government schemes policies subsidies and financial support for farmers."""
    try:
        if embeddings:
            query_embedding = embeddings.embed_query(query)
            results = schemes_collection.query(
                query_embeddings=[query_embedding],
                n_results=2
            )
        else:
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
            context = "\n\n".join([doc.page_content for doc in results['documents'][0]])
            formatted_result += f"### . Protection Method\n{context}\n\n"
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
        # Check if collection has any documents first
        if chat_history_collection.count() == 0:
            return []
        
        # Use get() method instead of query() for metadata filtering
        all_results = chat_history_collection.get()
        
        # Filter by session_id manually
        messages = []
        if all_results['metadatas']:
            session_data = []
            for i, metadata in enumerate(all_results['metadatas']):
                if metadata.get('session_id') == session_id:
                    session_data.append({
                        'document': all_results['documents'][i],
                        'metadata': metadata
                    })
            
            # Sort by timestamp
            session_data.sort(key=lambda x: x['metadata']['timestamp'])
            
            for item in session_data:
                doc = item['document']
                metadata = item['metadata']
                if metadata['type'] == 'HumanMessage':
                    messages.append(HumanMessage(content=doc))
                elif metadata['type'] == 'AIMessage':
                    messages.append(AIMessage(content=doc))
        
        return messages
    except Exception as e:
        print(f"Error loading chat history: {e}")
        return []

# --- Define the Enhanced LangGraph State ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    session_id: str
    next_action: str
    tool_results: List[str]
    selected_worker: str

# --- Supervisor Agent Node ---

def supervisor_agent(state: AgentState):
    """Supervisor agent that decides which worker to route the query to."""
    try:
        ollama_model = Ollama(model="llama3.2:3b")
        
        last_message = state["messages"][-1].content.lower()
        
        # Create worker options with their capabilities
        worker_options = """
Available Workers:
1. weather_worker - Handles weather, climate, temperature, rainfall, seasonal information
2. market_worker - Handles mandi prices, market rates, commodity prices, trading information  
3. scheme_worker - Handles government schemes, policies, subsidies, financial support
4. protection_worker - Handles crop protection, farming practices, organic methods, sustainable agriculture
5. general_worker - Handles general farming advice, techniques, and other agricultural queries
"""
        
        routing_prompt = f"""You are a supervisor agent for an agricultural advisory system. Based on the user's query, determine which specialized worker agent should handle this request.

{worker_options}

User Query: {last_message}

Analyze the query and respond with ONLY the worker name (e.g., "weather_worker", "market_worker", etc.) that best matches the user's needs. Consider keywords, context, and intent.

Selected Worker:"""
        
        response = ollama_model.invoke(routing_prompt)
        selected_worker = response.strip().lower()
        
        # Validate worker selection
        valid_workers = ["weather_worker", "market_worker", "scheme_worker", "protection_worker", "general_worker"]
        if selected_worker not in valid_workers:
            selected_worker = "general_worker"
        
        return {
            "selected_worker": selected_worker,
            "next_action": "route_to_worker"
        }
    
    except Exception as e:
        return {
            "selected_worker": "general_worker",
            "next_action": "route_to_worker"
        }

# --- Worker Agent Nodes ---

def weather_worker(state: AgentState):
    """Weather specialist worker agent."""
    last_message = state["messages"][-1]
    
    # Call weather tool
    tool_call = {
        "id": f"call_weather_{len(state['messages'])}",
        "name": "get_weather",
        "args": {}
    }
    
    ai_message = AIMessage(content="Fetching weather information for you...", tool_calls=[tool_call])
    return {"messages": [ai_message], "next_action": "call_tools"}

def market_worker(state: AgentState):
    """Market prices specialist worker agent."""
    last_message = state["messages"][-1]
    
    tool_call = {
        "id": f"call_mandi_{len(state['messages'])}",
        "name": "get_mandi_prices", 
        "args": {}
    }
    
    ai_message = AIMessage(content="Getting latest mandi prices for you...", tool_calls=[tool_call])
    return {"messages": [ai_message], "next_action": "call_tools"}

def scheme_worker(state: AgentState):
    """Government schemes specialist worker agent."""
    last_message = state["messages"][-1]
    
    tool_call = {
        "id": f"call_schemes_{len(state['messages'])}",
        "name": "search_schemes",
        "args": {"query": last_message.content}
    }
    
    ai_message = AIMessage(content="Searching government schemes information...", tool_calls=[tool_call])
    return {"messages": [ai_message], "next_action": "call_tools"}

def protection_worker(state: AgentState):
    """Crop protection specialist worker agent."""
    last_message = state["messages"][-1]
    
    tool_call = {
        "id": f"call_protection_{len(state['messages'])}",
        "name": "search_protection",
        "args": {"query": last_message.content}
    }
    
    ai_message = AIMessage(content="Looking up crop protection methods...", tool_calls=[tool_call])
    return {"messages": [ai_message], "next_action": "call_tools"}



def general_worker(state: AgentState):
    """General agricultural advice worker agent."""
    try:
        ollama_model = Ollama(model="llama3.2:3b")
        
        recent_messages = state["messages"][-5:]
        context = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in recent_messages])
        
        prompt = f"""You are a general agricultural advisor helping Indian farmers. Provide helpful, practical advice in a friendly tone using markdown formatting.

Use these formatting guidelines:
- Use ## for main headings
- Use ### for sub-headings  
- Use **bold** for important points
- Use bullet points with - for lists
- Use numbered lists when showing steps
- Use tables when comparing information
- Use emojis sparingly for visual appeal
- Answer in Hindi language when appropriate

Previous conversation context:
{context}

Current question: {state["messages"][-1].content}

Provide a well-formatted response with practical farming advice:"""
        
        response = ollama_model.invoke(prompt)
        ai_message = AIMessage(content=response)
        return {"messages": [ai_message], "next_action": "end"}
    
    except Exception as e:
        error_message = AIMessage(content=f"I'm having trouble connecting to my knowledge base. Please ensure the AI service is running. Error: {str(e)}")
        return {"messages": [error_message], "next_action": "end"}

# --- Tool Processing and Final Response Node ---

def process_tool_results(state: AgentState):
    """Process tool results and generate final formatted response."""
    try:
        ollama_model = Ollama(model="llama3.2:3b")
        
        # Get the tool results from the last tool message
        tool_results = []
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                tool_results.append(msg.content)
        
        if not tool_results:
            return {"messages": [AIMessage(content="No tool results found.")], "next_action": "end"}
        
        # Combine tool results
        combined_results = "\n\n".join(tool_results)
        original_query = None
        
        # Find the original human message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                original_query = msg.content
                break
        
        formatting_prompt = f"""You are an expert agricultural advisor. You have received information from specialized tools. Your task is to format this information into a comprehensive, well-structured response for the farmer.

Original Query: {original_query}

Tool Results:
{combined_results}

Please format the response with:
- Clear headings using ## and ###
- **Bold** text for important points
- Bullet points for lists
- Tables when appropriate
- Emojis for visual appeal
- Practical actionable advice
- Professional but friendly tone
- Always Answer in Hindi language when appropriate

Create a comprehensive response that directly addresses the farmer's query using the tool information:"""
        
        formatted_response = ollama_model.invoke(formatting_prompt)
        ai_message = AIMessage(content=formatted_response)
        
        return {"messages": [ai_message], "next_action": "end"}
    
    except Exception as e:
        error_message = AIMessage(content=f"Error processing results: {str(e)}")
        return {"messages": [error_message], "next_action": "end"}

# --- Router Functions ---

def route_to_worker(state: AgentState):
    """Route to the appropriate worker based on supervisor decision."""
    selected_worker = state.get("selected_worker", "general_worker")
    return selected_worker

def should_continue_from_supervisor(state: AgentState):
    """Determine the next step from supervisor."""
    next_action = state.get("next_action", "end")
    if next_action == "route_to_worker":
        return route_to_worker(state)
    else:
        return "end"

def should_continue_from_worker(state: AgentState):
    """Determine the next step from worker nodes."""
    next_action = state.get("next_action", "end")
    # ALL tool-using workers should go to tools, then to process_results
    if next_action == "call_tools":
        return "tools"
    else:
        return "end"

def should_continue_from_results(state: AgentState):
    """Determine the next step from process results."""
    return "end"

# --- Workflow Visualization Function ---
def visualize_workflow():
    """Generate a visual representation of the workflow."""
    try:
        # Create a simple ASCII visualization
        visualization = """
🏗️ **Multi-Agent Workflow Architecture**

┌─────────────────┐
│   User Query    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Supervisor    │ ◄─── Analyzes query & selects worker
│     Agent       │
└─────────┬───────┘
          │
          ▼
    ┌─────────────┐
    │   Router    │
    └──────┬──────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │                                             │
    ▼              ▼         ▼         ▼         ▼
┌─────────┐   ┌─────────┐ ┌─────┐ ┌─────────┐ ┌─────┐
│Weather  │   │Market   │ │Pest │ │Protection│ │Scheme│
│Worker   │   │Worker   │ │Work.│ │Worker   │ │Work.│
└────┬────┘   └────┬────┘ └──┬──┘ └────┬────┘ └──┬──┘
     │             │         │         │         │
     └─────────────┼─────────┼─────────┼─────────┘
                   │         │         │
                   ▼         ▼         ▼
              ┌─────────────────────────┐
              │     Tools Execution     │ ◄─── Calls appropriate tools
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Process Results       │ ◄─── LLM processes tool outputs
              │   (Final LLM)           │
              └───────────┬─────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │    Final Response       │
              └─────────────────────────┘

🔄 **Flow Summary:**
1. User query → Supervisor analyzes intent
2. Supervisor → Routes to specialized worker
3. Worker → Calls appropriate tools
4. Tools → Execute and return raw data
5. Process Results → LLM formats and enhances response
6. Final Response → Delivered to user
        """
        return visualization
    except Exception as e:
        return f"Error generating visualization: {e}"

# --- Build the Enhanced Multi-Agent LangGraph ---

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("weather_worker", weather_worker)
workflow.add_node("market_worker", market_worker) 
workflow.add_node("scheme_worker", scheme_worker)
workflow.add_node("protection_worker", protection_worker)
workflow.add_node("general_worker", general_worker)
workflow.add_node("tools", tool_node)
workflow.add_node("process_results", process_tool_results)

# Set entry point
workflow.set_entry_point("supervisor")

# Add conditional edges from supervisor to workers
workflow.add_conditional_edges(
    "supervisor",
    should_continue_from_supervisor,
    {
        "weather_worker": "weather_worker",
        "market_worker": "market_worker", 
        "scheme_worker": "scheme_worker",
        "protection_worker": "protection_worker",
        "general_worker": "general_worker",
        "end": END
    }
)

# CRITICAL: All tool-using workers MUST go through tools -> process_results
# Add conditional edges from tool-using workers - they ALL go to tools
for worker in ["weather_worker", "market_worker", "scheme_worker", "protection_worker"]: 
    workflow.add_conditional_edges(
        worker,
        should_continue_from_worker,
        {
            "tools": "tools"
        }
    )

# General worker goes directly to end (no tools needed)
workflow.add_edge("general_worker", END)

# CRITICAL: Tools node ALWAYS goes to process_results (no conditional)
workflow.add_edge("tools", "process_results")

# Process results ALWAYS goes to end (final response)
workflow.add_edge("process_results", END)

# Compile the graph
app_graph = workflow.compile()





# Visualize as PNG
from PIL import Image
from io import BytesIO

png_data = app_graph.get_graph().draw_png()
arch = Image.open(BytesIO(png_data))
arch.save("workflow.png")
#import pdb; pdb.set_trace()
# --- FastAPI Models ---

class ChatMessage(BaseModel):
    message: str
    session_id: str = None

# --- FastAPI Endpoints ---

@app.get("/visualize")
async def visualize_workflow_endpoint():
    """Endpoint to visualize the workflow structure."""
    return {
        "visualization": visualize_workflow(),
        "workflow_summary": {
            "total_nodes": 9,
            "entry_point": "supervisor", 
            "workers": ["weather_worker", "market_worker", "scheme_worker", "protection_worker", "general_worker"],
            "tools_available": [tool.name for tool in tools],
            "processing_flow": "supervisor → worker → tools → process_results → end"
        }
    }

@app.get("/debug/{session_id}")
async def debug_chat_session(session_id: str):
    """Debug endpoint to inspect chat session state."""
    try:
        session_messages = chat_sessions.get(session_id, [])
        history_from_db = load_chat_history(session_id)
        
        return {
            "session_id": session_id,
            "messages_in_memory": len(session_messages),
            "messages_in_db": len(history_from_db),
            "recent_messages": [
                {"type": type(msg).__name__, "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content}
                for msg in session_messages[-5:]
            ],
            "db_connection": "healthy" if chat_history_collection.count() >= 0 else "error"
        }
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """Main chat endpoint with multi-agent memory support."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Initialize session if not exists
        if session_id not in chat_sessions:
            chat_sessions[session_id] = load_chat_history(session_id)
        
        user_message = HumanMessage(content=request.message)
        chat_sessions[session_id].append(user_message)
        save_chat_history(session_id, user_message)
        
        state = {
            "messages": chat_sessions[session_id],
            "session_id": session_id,
            "next_action": "",
            "tool_results": [],
            "selected_worker": ""
        }
        
        # Execute the workflow
        result = app_graph.invoke(state)
        
        # Get the final AI response
        ai_response = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                ai_response = msg
                break
        
        if not ai_response:
            ai_response = AIMessage(content="I apologize, but I couldn't generate a proper response. Please try again.")
        
        # Update session history with all new messages
        chat_sessions[session_id] = result["messages"]
        save_chat_history(session_id, ai_response)
        
        return {
            "response": ai_response.content,
            "session_id": session_id,
            "worker_used": result.get("selected_worker", "unknown"),
            "total_messages": len(result["messages"]),
            "processing_path": "supervisor → worker → tools → llm_processing → response"
        }
    
    except Exception as e:
        error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
        return {
            "response": error_response,
            "session_id": request.session_id or str(uuid.uuid4()),
            "worker_used": "error",
            "processing_path": "error_handling"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.delete("/chat/{session_id}")
async def clear_chat_history(session_id: str):
    """Clear chat history for a specific session."""
    try:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        chat_history_collection.delete(where={"session_id": session_id})
        
        return {"message": "Chat history cleared successfully"}
    except Exception as e:
        return {"error": f"Failed to clear chat history: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("krishi_mitra:app", host="0.0.0.0", port=8000, reload=True)