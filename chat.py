# backend.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
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

# Initialize collections
schemes_collection = chroma_client.get_or_create_collection("agri_schemes")
protection_collection = chroma_client.get_or_create_collection("protection")
pest_collection = chroma_client.get_or_create_collection("pest")
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
    
    pest_data = [
        {
            "id": "pest_1",
            "text": "Aphids are small soft-bodied insects that suck plant sap. Control methods include introducing natural predators like ladybugs, using neem oil spray, or applying insecticidal soap.",
            "metadata": {"pest": "aphids", "control": "biological"}
        },
        {
            "id": "pest_2",
            "text": "Whiteflies are tiny flying insects that damage plants by sucking sap and transmitting viruses. Management includes yellow sticky traps, reflective mulches, and insecticidal soaps.",
            "metadata": {"pest": "whiteflies", "control": "physical"}
        },
        {
            "id": "pest_3",
            "text": "Armyworms are caterpillars that can cause severe damage to crops. Control methods include hand-picking in small areas, using pheromone traps, or applying Bacillus thuringiensis (Bt).",
            "metadata": {"pest": "armyworms", "control": "biological"}
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
        
        if pest_collection.count() == 0:
            for item in pest_data:
                if embeddings:
                    embedding = embeddings.embed_query(item["text"])
                    pest_collection.add(
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
| **Onion** | Nashik | ₹3,200 | ↔️ Stable |
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
    """Search for crop protection and farming best practices information."""
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
            for i, doc in enumerate(results['documents'][0]):
                formatted_result += f"### {i+1}. Protection Method\n{doc}\n\n"
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
                n_results=2
            )
        else:
            results = pest_collection.query(
                query_texts=[query],
                n_results=2
            )
        
        if results['documents'][0]:
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

# --- Define the LangGraph State (using MessagesState pattern) ---
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    session_id: str

# --- Supervisor Node Functions ---

# Initialize Ollama model
try:
    supervisor_model = Ollama(model="llama3.2:3b")
except:
    print("Warning: Could not initialize Ollama model")
    supervisor_model = None

def generate_query_or_respond(state: AgentState):
    """
    Supervisor node that decides whether to use tools or respond directly.
    This is the main decision-making node based on the agentic RAG pattern.
    """
    try:
        if not supervisor_model:
            raise Exception("Supervisor model not initialized")
        
        # Get recent context from messages
        messages = state["messages"]
        current_question = messages[-1].content if messages else ""
        
        # Create a prompt that helps the model decide when to use tools
        system_prompt = """You are an expert agricultural advisor helping Indian farmers. 
        
You have access to these tools:
- get_weather: Use when asked about weather, climate, temperature, rain, mausam
- get_mandi_prices: Use when asked about market prices, mandi rates, bhav, costs
- search_schemes: Use when asked about government schemes, yojana, policies
- search_protection: Use when asked about crop protection, plant care, cultivation practices
- search_pest_info: Use when asked about pests, insects, diseases, bugs

Analyze the user's question and decide whether to:
1. Use relevant tools to get specific information
2. Respond directly with general agricultural advice

Current conversation context: {context}

User question: {question}

If you need to use tools, call the appropriate ones. Otherwise, provide helpful agricultural advice in Hindi with markdown formatting."""

        context = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages[-3:]])
        
        prompt = system_prompt.format(context=context, question=current_question)
        
        # Bind tools to the model and invoke
        response = supervisor_model.bind_tools(tools).invoke([{"role": "user", "content": prompt}])
        
        # Convert Ollama response to LangChain AIMessage format
        if hasattr(response, 'tool_calls') and response.tool_calls:
            ai_message = AIMessage(content=response.content or "", tool_calls=response.tool_calls)
        else:
            ai_message = AIMessage(content=response if isinstance(response, str) else str(response))
        
        return {"messages": [ai_message]}
        
    except Exception as e:
        # Fallback to direct response
        fallback_prompt = f"""You are an expert agricultural advisor. Answer this farming question in Hindi with helpful, practical advice using markdown formatting:

Question: {current_question}

Provide a well-formatted response with practical farming advice."""
        
        try:
            response = supervisor_model.invoke(fallback_prompt)
            ai_message = AIMessage(content=response if isinstance(response, str) else str(response))
            return {"messages": [ai_message]}
        except:
            error_message = AIMessage(content="मुझे खुशी होगी आपकी मदद करने में, लेकिन अभी मैं अपनी सेवाओं से जुड़ने में कुछ समस्या का सामना कर रहा हूँ। कृपया दोबारा कोशिश करें।")
            return {"messages": [error_message]}

# Document grading for quality control
class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""
    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )

def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """
    Determine whether the retrieved documents are relevant to the question.
    This implements the quality control mechanism from the agentic RAG pattern.
    """
    try:
        if len(state["messages"]) < 2:
            return "generate_answer"
            
        question = state["messages"][0].content
        # Get the last tool message content
        last_message = state["messages"][-1]
        
        if hasattr(last_message, 'content'):
            context = last_message.content
        else:
            return "generate_answer"
        
        # Simple relevance check - in production, use a more sophisticated model
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Check for keyword overlap or meaningful content
        common_keywords = ["scheme", "yojana", "weather", "mandi", "price", "pest", "protection", "crop", "farming"]
        
        has_relevant_content = any(keyword in context_lower for keyword in common_keywords)
        has_meaningful_content = len(context) > 50 and "error" not in context_lower
        
        if has_relevant_content and has_meaningful_content:
            return "generate_answer"
        else:
            return "rewrite_question"
            
    except Exception as e:
        print(f"Error in grade_documents: {e}")
        return "generate_answer"

def rewrite_question(state: AgentState):
    """
    Rewrite the original user question to improve retrieval results.
    This implements the query refinement mechanism from the agentic RAG pattern.
    """
    try:
        messages = state["messages"]
        original_question = messages[0].content
        
        rewrite_prompt = f"""Look at this farming question and rewrite it to be more specific and clear for better information retrieval:

Original question: {original_question}

Rewrite this question to be more specific about:
- What type of farming information is needed
- Specific crops or farming practices if mentioned
- Geographic or seasonal context if relevant

Improved question:"""

        if supervisor_model:
            response = supervisor_model.invoke(rewrite_prompt)
            rewritten_content = response if isinstance(response, str) else str(response)
            return {"messages": [HumanMessage(content=rewritten_content)]}
        else:
            # Fallback: return original question
            return {"messages": [HumanMessage(content=original_question)]}
            
    except Exception as e:
        print(f"Error in rewrite_question: {e}")
        # Return original question on error
        return {"messages": [HumanMessage(content=state["messages"][0].content)]}

def generate_answer(state: AgentState):
    """
    Generate final answer using the retrieved context.
    This implements the answer generation from the agentic RAG pattern.
    """
    try:
        question = state["messages"][0].content
        context = state["messages"][-1].content
        
        answer_prompt = f"""You are an expert agricultural advisor. Use the following retrieved information to answer the farmer's question in Hindi with practical, helpful advice.

Question: {question}

Retrieved Information: {context}

Instructions:
- Provide practical, actionable advice
- Use markdown formatting for better readability
- Keep the response comprehensive but concise
- Answer in Hindi language
- Include specific recommendations where possible

Answer:"""

        if supervisor_model:
            response = supervisor_model.invoke(answer_prompt)
            ai_message = AIMessage(content=response if isinstance(response, str) else str(response))
            return {"messages": [ai_message]}
        else:
            ai_message = AIMessage(content="मुझे खुशी होगी आपकी मदद करने में, लेकिन अभी सिस्टम में कुछ समस्या है।")
            return {"messages": [ai_message]}
            
    except Exception as e:
        print(f"Error in generate_answer: {e}")
        error_message = AIMessage(content=f"उत्तर तैयार करने में समस्या हुई: {str(e)}")
        return {"messages": [error_message]}

# --- Build the Supervisor LangGraph ---

workflow = StateGraph(AgentState)

# Add nodes following the agentic RAG pattern
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", tool_node)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# Set entry point
workflow.add_edge(START, "generate_query_or_respond")

# Add conditional edges for tool usage decision
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Add conditional edges for document grading
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
    }
)

# Connect final nodes
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile the graph
app_graph = workflow.compile()

# Visualize as PNG
from PIL import Image
from io import BytesIO

png_data = app_graph.get_graph().draw_png()
arch = Image.open(BytesIO(png_data))
arch.save("workflow.png")
# --- FastAPI Models ---

class ChatMessage(BaseModel):
    message: str
    session_id: str = None

# --- FastAPI Endpoints ---

@app.post("/chat")
async def chat_endpoint(request: ChatMessage):
    """Main chat endpoint with supervisor pattern and memory support."""
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
        
        # Prepare state for the supervisor graph
        state = AgentState(
            messages=chat_sessions[session_id],
            session_id=session_id
        )
        
        # Run through the supervisor graph
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
        error_response = f"मुझे खुशी होगी आपकी मदद करने में, लेकिन कुछ तकनीकी समस्या का सामना हो रहा है: {str(e)}। कृपया दोबारा कोशिश करें।"
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