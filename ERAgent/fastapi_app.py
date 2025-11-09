"""
FastAPI application for ER Patient Vital Monitoring Assistant
Provides a custom web UI for the ADK agent
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

# Import the agent
from ER_Vitals_Monitoring_Assistant.agent import main_agent

# Initialize FastAPI app
app = FastAPI(
    title="ER Patient Vital Monitoring",
    description="AI-powered patient vital monitoring and analysis system",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    summary: str
    matches: int
    analysis: Dict[str, Any]
    effects_analysis: Dict[str, Any]
    pdf_report_url: Optional[str] = None
    error: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ER Vital Monitoring"}

@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Query the ER monitoring agent
    
    Args:
        request: QueryRequest with user query and optional top_k
        
    Returns:
        QueryResponse with analysis results
    """
    try:
        print(f"📥 Received query: {request.query}")
        
        # Call the main agent
        response = main_agent.handle_query(request.query, top_k=request.top_k)
        
        # Format response
        result = QueryResponse(
            query=response.get("query", request.query),
            summary=response.get("summary", "No summary available"),
            matches=response.get("matches", 0),
            analysis=response.get("analysis", {}),
            effects_analysis=response.get("effects_analysis", {}),
            pdf_report_url=response.get("pdf_report_url"),
            error=response.get("error")
        )
        
        print(f"✅ Query processed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/welcome")
async def get_welcome_message():
    """Get the welcome message for the agent"""
    return {
        "message": """👋 Hello! I'm the ER Patient Vital Monitoring Assistant for City Hospital.

I can help you with:
🔍 Query patient vital signs and readings
📊 Analyze patient data and trends
🏥 Identify potential medical conditions based on vitals
📈 Generate visual graphs of patient vitals over time
📄 Create comprehensive PDF reports with analysis

You can ask me questions like:
• 'Show me the latest readings for patient P100'
• 'What are the potential effects of these readings?'
• 'Generate a PDF report for patient P105'


How can I assist you today?"""
    }

# Mount static files (will create this folder next)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    print("🚀 Starting ER Patient Vital Monitoring Assistant...")
    print("📍 Access the application at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

