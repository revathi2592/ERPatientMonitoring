# ER Patient Vital Monitoring - Multi-Agent Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────┐              ┌──────────────────────┐            │
│  │   FastAPI Web UI     │              │   ADK Web Interface  │            │
│  │  (Custom Frontend)   │              │  (Default Google)    │            │
│  │                      │              │                      │            │
│  │  - index.html        │              │  - ADK Web Server    │            │
│  │  - style.css         │              │  - Default UI        │            │
│  │  - script.js         │              │                      │            │
│  └──────────┬───────────┘              └──────────┬───────────┘            │
│             │                                     │                         │
│             │ HTTP POST /api/query                │ ADK Protocol            │
│             │                                     │                         │
└─────────────┼─────────────────────────────────────┼─────────────────────────┘
              │                                     │
              ▼                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    ROOT AGENT (ADK Agent)                           │     │
│  │                 ERMonitoringAgent (BaseAgent)                       │     │
│  │                                                                      │     │
│  │  - Model: gemini-2.0-flash                                          │     │
│  │  - Tool: adk_tool_handle_query()                                    │     │
│  │  - Orchestrates conversation flow                                   │     │
│  │  - Invokes MainAgent via tool                                       │     │
│  └──────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                            │
│                                 │ Calls                                      │
│                                 ▼                                            │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                         MAIN AGENT                                  │     │
│  │                     (Orchestration Layer)                           │     │
│  │                                                                      │     │
│  │  Responsibilities:                                                  │     │
│  │  - Query routing and contextual enhancement                         │     │
│  │  - Multi-agent coordination                                         │     │
│  │  - Response aggregation                                             │     │
│  │  - PDF generation decision logic                                    │     │
│  │                                                                      │     │
│  │  Components:                                                        │     │
│  │  ┌──────────────┬──────────────┬──────────────┬──────────────┐     │     │
│  │  │   Retrieval  │   Analysis   │   Effects    │    Report    │     │     │
│  │  │     Agent    │     Agent    │   Analysis   │     Agent    │     │     │
│  │  │              │              │     Agent    │              │     │     │
│  │  └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘     │     │
│  │         │              │              │              │              │     │
│  └─────────┼──────────────┼──────────────┼──────────────┼──────────────┘     │
│            │              │              │              │                    │
└────────────┼──────────────┼──────────────┼──────────────┼────────────────────┘
             │              │              │              │
             │              │              │              │
┌────────────┼──────────────┼──────────────┼──────────────┼────────────────────┐
│            │              │              │              │                     │
│    SPECIALIZED AGENT LAYER (5 Agents)                                        │
├────────────┼──────────────┼──────────────┼──────────────┼────────────────────┤
│            ▼              ▼              ▼              ▼                     │
│  ┌─────────────────┐  ┌─────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │ 1. RETRIEVAL    │  │ 2. ANALYSIS │  │ 3. EFFECTS       │  │ 4. REPORT  │ │
│  │    AGENT        │  │    AGENT    │  │    ANALYSIS      │  │    AGENT   │ │
│  │                 │  │             │  │    AGENT         │  │            │ │
│  │ Purpose:        │  │ Purpose:    │  │                  │  │ Purpose:   │ │
│  │ - Query         │  │ - Stats     │  │ Purpose:         │  │ - PDF      │ │
│  │   classification│  │   calculation│  │ - Condition     │  │   generation│
│  │ - SQL/Embedding │  │ - Aggregate │  │   detection     │  │ - Graph    │ │
│  │   routing       │  │   metrics   │  │ - Effects       │  │   creation │ │
│  │ - BigQuery      │  │             │  │   matching      │  │ - Report   │ │
│  │   search        │  │ Outputs:    │  │                  │  │   formatting│
│  │ - LLM summary   │  │ - Mean/Min/ │  │ Methods:        │  │            │ │
│  │                 │  │   Max/Std   │  │ - Query effects │  │ Outputs:   │ │
│  │ Methods:        │  │ - Row count │  │   master table  │  │ - PDF file │ │
│  │ - search()      │  │             │  │ - Match vitals  │  │   path     │ │
│  │ - summarize()   │  │             │  │   to ranges     │  │ - 3-page   │ │
│  │                 │  │             │  │                  │  │   report   │ │
│  │ Uses:           │  │             │  │ Outputs:        │  │            │ │
│  │ - Gemini 2.0    │  │             │  │ - Conditions    │  │ Uses:      │ │
│  │ - Text          │  │             │  │   list          │  │ - ReportLab│ │
│  │   Embeddings    │  │             │  │ - Potential     │  │ - Matplotlib│
│  └────────┬────────┘  └─────────────┘  │   effects       │  └─────┬──────┘ │
│           │                             │ - Vitals        │        │        │
│           │                             │   analyzed      │        │        │
│           │                             └──────────────────┘        │        │
│           │                                                         │        │
│           │                             ┌──────────────────┐        │        │
│           │                             │ 5. UPLOADER      │        │        │
│           │                             │    AGENT         │◄───────┘        │
│           │                             │                  │                 │
│           │                             │ Purpose:         │                 │
│           │                             │ - GCS upload     │                 │
│           │                             │ - URL generation │                 │
│           │                             │                  │                 │
│           │                             │ Methods:         │                 │
│           │                             │ - upload_to_gcs()│                 │
│           │                             │                  │                 │
│           │                             │ Features:        │                 │
│           │                             │ - Signed URLs    │                 │
│           │                             │   (local)        │                 │
│           │                             │ - Public URLs    │                 │
│           │                             │   (Cloud Run)    │                 │
│           │                             └──────────────────┘                 │
│           │                                                                   │
└───────────┼───────────────────────────────────────────────────────────────────┘
            │
            │
┌───────────┼───────────────────────────────────────────────────────────────────┐
│           │          SUPPORTING COMPONENTS LAYER                              │
├───────────┼───────────────────────────────────────────────────────────────────┤
│           ▼                                                                   │
│  ┌─────────────────────┐        ┌──────────────────────┐                    │
│  │  MEMORY MANAGER     │        │  HELPER FUNCTIONS     │                    │
│  │  (Contextual State) │        │                       │                    │
│  │                     │        │ - classify_query_type()│                   │
│  │ Purpose:            │        │   (Rule-based + LLM)  │                    │
│  │ - Multi-turn context│        │                       │                    │
│  │ - Patient ID        │        │ - generate_sql_from_nl()│                  │
│  │   tracking          │        │   (LLM SQL generation)│                    │
│  │ - Conversation      │        │                       │                    │
│  │   history           │        │ - get_query_embedding()│                   │
│  │                     │        │   (Vertex AI)         │                    │
│  │ Methods:            │        │                       │                    │
│  │ - add_entry()       │        │ - bigquery_cosine_search()│                │
│  │ - get_last()        │        │                       │                    │
│  │ - get_contextual_   │        │ - rows_to_dataframe() │                    │
│  │   query()           │        │                       │                    │
│  │                     │        │ - draw_vitals_graph() │                    │
│  │ Features:           │        │                       │                    │
│  │ - Last 5 queries    │        │ - build_pdf_report_   │                    │
│  │ - Patient context   │        │   with_graphs()       │                    │
│  │   injection         │        │                       │                    │
│  └─────────────────────┘        └──────────────────────┘                     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


┌───────────────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL SERVICES LAYER                                │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   BigQuery   │  │  Vertex AI   │  │     GCS      │  │  Effects Master │  │
│  │              │  │              │  │   Storage    │  │     Table       │  │
│  │ Tables:      │  │ Models:      │  │              │  │                 │  │
│  │ - patient_   │  │ - Gemini 2.0 │  │ Bucket:      │  │ Schema:         │  │
│  │   vitals     │  │   Flash      │  │ - erpatient  │  │ - condition     │  │
│  │              │  │   (LLM)      │  │   vitals     │  │ - heart_rate_   │  │
│  │ Schema:      │  │              │  │              │  │   min/max       │  │
│  │ - patient_id │  │ - text-      │  │ Storage:     │  │ - bp_systolic_  │  │
│  │ - timestamp  │  │   embedding- │  │ - PDF reports│  │   min/max       │  │
│  │ - heart_rate │  │   004        │  │ - Graphs     │  │ - bp_diastolic_ │  │
│  │ - bp_systolic│  │   (Embeddings│  │              │  │   min/max       │  │
│  │ - bp_        │  │              │  │ Features:    │  │ - oxygen_level_ │  │
│  │   diastolic  │  │              │  │ - Public     │  │   min/max       │  │
│  │ - oxygen_    │  │              │  │   access     │  │ - potential_    │  │
│  │   level      │  │              │  │ - Signed URLs│  │   effects       │  │
│  │ - embedding  │  │              │  │              │  │                 │  │
│  │   (ARRAY)    │  │              │  │              │  │                 │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘  │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Agent Flow Diagram

```
User Query: "Show me the latest readings for patient P100"
     │
     ▼
┌─────────────────────────────────────┐
│  ROOT AGENT (ERMonitoringAgent)    │
│  - Receives user query              │
│  - Calls adk_tool_handle_query()    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  MAIN AGENT                         │
│  - handle_query()                   │
│  - Store original query             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  MEMORY MANAGER                     │
│  - get_contextual_query()           │
│  - Inject patient ID if contextual  │
│  Output: Enhanced query             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  RETRIEVAL AGENT                    │
│  Step 1: classify_query_type()      │
│    - Rule-based: "latest" → SQL     │
│    - Fallback: LLM classification   │
│                                     │
│  Step 2: Generate & Execute         │
│    - generate_sql_from_nl() (LLM)   │
│    - execute_bq_sql()               │
│                                     │
│  Step 3: Summarize                  │
│    - summarize_results() (LLM)      │
│                                     │
│  Output: {summary, rows}            │
└────────────┬────────────────────────┘
             │
             ├──────┬──────────────────┬────────────────────┐
             ▼      ▼                  ▼                    ▼
      ┌──────────┐ ┌──────────┐ ┌──────────────┐  ┌──────────────┐
      │ ANALYSIS │ │ EFFECTS  │ │ MEMORY       │  │ PDF?         │
      │ AGENT    │ │ ANALYSIS │ │ MANAGER      │  │ (Conditional)│
      │          │ │ AGENT    │ │              │  │              │
      │ analyze()│ │ analyze_ │ │ add_entry()  │  │ If "pdf" in  │
      │          │ │ patient_ │ │              │  │ query:       │
      │ Output:  │ │ vitals() │ │ Store:       │  │              │
      │ - stats  │ │          │ │ - query      │  │ ┌──────────┐ │
      │ - mean/  │ │ Query    │ │ - rows       │  │ │  REPORT  │ │
      │   min/   │ │ effects_ │ │ - analysis   │  │ │  AGENT   │ │
      │   max/   │ │ master_  │ │ - patient_id │  │ │          │ │
      │   std    │ │ table    │ │              │  │ │ create_  │ │
      │          │ │          │ │              │  │ │ pdf()    │ │
      │          │ │ Output:  │ │              │  │ │          │ │
      │          │ │ - vitals_│ │              │  │ │ Output:  │ │
      │          │ │   analyzed│ │              │  │ │ - PDF    │ │
      │          │ │ - conditions│ │            │  │ │   path   │ │
      │          │ │ - condition_│ │            │  │ │          │ │
      │          │ │   count  │ │              │  │ └────┬─────┘ │
      └──────────┘ └──────────┘ └──────────────┘  │      │       │
                                                   │      ▼       │
                                                   │ ┌──────────┐ │
                                                   │ │ UPLOADER │ │
                                                   │ │  AGENT   │ │
                                                   │ │          │ │
                                                   │ │ upload_  │ │
                                                   │ │ to_gcs() │ │
                                                   │ │          │ │
                                                   │ │ Output:  │ │
                                                   │ │ - URL    │ │
                                                   │ └──────────┘ │
                                                   └──────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  MAIN AGENT                │
                    │  - Aggregate all results   │
                    │  - Build final response    │
                    │                            │
                    │  Response:                 │
                    │  {                         │
                    │    query,                  │
                    │    summary,                │
                    │    matches,                │
                    │    analysis,               │
                    │    effects_analysis,       │
                    │    pdf_report_url          │
                    │  }                         │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  ROOT AGENT                │
                    │  - to_serializable()       │
                    │  - Return to user          │
                    └────────────┬───────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────┐
                    │  USER INTERFACE            │
                    │  - Display results         │
                    │  - Show effects analysis   │
                    │  - PDF download link       │
                    └────────────────────────────┘
```

## Key Design Patterns

### 1. **Multi-Agent Orchestration**
- **MainAgent** coordinates 5 specialized agents
- Each agent has a single responsibility
- Agents can be composed and extended independently

### 2. **Contextual Memory**
- **MemoryManager** maintains conversation state
- Tracks last patient ID for contextual queries
- Enables multi-turn conversations

### 3. **Hybrid Retrieval**
- **Rule-based classification** for common queries (fast)
- **LLM-based fallback** for ambiguous queries (accurate)
- Supports both SQL and embedding-based search

### 4. **Conditional Workflows**
- PDF generation only when requested
- Effects analysis always runs (for health insights)
- Contextual query enhancement based on history

### 5. **LLM Integration Points**
1. Query classification (Gemini 2.0 Flash)
2. SQL generation from natural language (Gemini 2.0 Flash)
3. Result summarization (Gemini 2.0 Flash)
4. Embeddings generation (text-embedding-004)

## Agent Responsibilities Summary

| Agent | Purpose | Input | Output | External Services |
|-------|---------|-------|--------|-------------------|
| **Root Agent** | User interaction orchestration | User query | Formatted response | Gemini 2.0 Flash |
| **Main Agent** | Multi-agent coordination | User query | Aggregated results | None (orchestrator) |
| **Retrieval Agent** | Data retrieval & search | Query string | Rows + summary | BigQuery, Vertex AI |
| **Analysis Agent** | Statistical analysis | Data rows | Stats (mean/min/max/std) | None |
| **Effects Analysis Agent** | Clinical condition detection | Vital signs | Conditions + effects | BigQuery (effects table) |
| **Report Agent** | PDF generation | Query + rows + effects | PDF file path | ReportLab, Matplotlib |
| **Uploader Agent** | Cloud storage | PDF file path | Download URL | GCS |
| **Memory Manager** | Contextual state | Query + results | Enhanced query | None |

## Technology Stack

- **Framework**: Google ADK (Agent Development Kit)
- **LLM**: Gemini 2.0 Flash
- **Embeddings**: text-embedding-004
- **Database**: BigQuery
- **Storage**: Google Cloud Storage
- **PDF**: ReportLab
- **Visualization**: Matplotlib
- **Web Interface**: FastAPI + HTML/CSS/JS
- **Deployment**: Cloud Run (containerized)

## Data Flow

1. **Query Processing**: User → FastAPI/ADK → Root Agent
2. **Retrieval**: Main Agent → Retrieval Agent → BigQuery
3. **Analysis**: Parallel execution of Analysis + Effects Analysis
4. **Memory**: Store query context for future turns
5. **PDF Generation**: Conditional (Report Agent → Uploader Agent → GCS)
6. **Response**: Aggregated results → User interface
