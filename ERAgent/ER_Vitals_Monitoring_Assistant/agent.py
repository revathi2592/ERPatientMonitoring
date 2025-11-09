import os
import json
import tempfile
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
 
# ADK imports
from google.adk import Agent
from google.adk.agents import BaseAgent
 
# GCP clients
from google.cloud import bigquery, storage
from google.cloud import aiplatform
import google.auth
 
# Vertex AI
import vertexai
from vertexai.language_models import TextEmbeddingModel
 
# PDF + plotting
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# data handling
import pandas as pd
import numpy as np

#Generative Model
from vertexai.generative_models import GenerativeModel
 
# ---------- CONFIG ----------
PROJECT_ID = os.environ.get("PROJECT_ID", "spiritual-clock-471207-i1")
BQ_DATASET = os.environ.get("BQ_DATASET", "your_dataset")
BQ_TABLE = os.environ.get("BQ_TABLE", "your_table")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-004")
TOP_K = int(os.environ.get("TOP_K", "5"))
VERTEX_LOCATION = os.environ.get("GOOGLE_VERTEXAI_LOCATION", "us-central1")
GCS_BUCKET = os.environ.get("GCS_BUCKET", "erpatientvitals")
 
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
os.environ["GOOGLE_VERTEXAI_PROJECT"] = PROJECT_ID
os.environ["GOOGLE_VERTEXAI_LOCATION"] = VERTEX_LOCATION

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
 
vertexai.init(project=PROJECT_ID, location=VERTEX_LOCATION)
bq_client = bigquery.Client(project=PROJECT_ID)
aiplatform.init(project=PROJECT_ID)
 
# ---------- Embedding ----------
def get_query_embedding(text: str) -> List[float]:
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
    response = model.get_embeddings([text])
    return list(response[0].values)
 
# ---------- BigQuery Search ----------
def bigquery_cosine_search(query_vector: List[float], k: int = TOP_K,
                           patient_id: Optional[str] = None, all_patients: bool = False) -> List[Dict[str, Any]]:
    q_arr_literal = "[" + ",".join(f"{float(x)}" for x in query_vector) + "]"
    where_clause = ""
    if patient_id:
        where_clause = f"WHERE patient_id = '{patient_id}'"
    elif all_patients:
        where_clause = "WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)"
 
    sql = f"""
    WITH q AS ( SELECT {q_arr_literal} AS qvec )
    SELECT
      t.* EXCEPT(embedding),
      (SELECT SUM(e * qv)
       FROM UNNEST(t.embedding) AS e WITH OFFSET idx
       JOIN UNNEST(q.qvec) AS qv WITH OFFSET jdx ON idx = jdx) AS dot,
      (SELECT SQRT(SUM(e*e)) FROM UNNEST(t.embedding) AS e) AS norm_t,
      (SELECT SQRT(SUM(qv*qv)) FROM UNNEST(q.qvec) AS qv) AS norm_q
    FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}` AS t, q
    {where_clause}
    ORDER BY dot / (norm_t * norm_q) DESC
    LIMIT {k}
    """
 
    rows = list(bq_client.query(sql).result())
    results = []
    for r in rows:
        rd = dict(r)
        try:
            sim = float(rd.get("dot") / (rd.get("norm_t") * rd.get("norm_q")))
        except Exception:
            sim = None
        rd["cosine_similarity"] = sim
        results.append(rd)
    return results
 
# ---------- Helpers ----------
def rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if df[col].dtype == object:
            def try_parse(x):
                if isinstance(x, str) and (x.startswith("{") or x.startswith("[")):
                    try:
                        return json.loads(x)
                    except Exception:
                        return x
                return x
            df[col] = df[col].apply(try_parse)
    for candidate in ["timestamp", "time", "created_at"]:
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate], errors="coerce")
    return df

# LLM to translate natural language to SQL
def generate_sql_from_nl(query_text: str) -> str:
    model = GenerativeModel("gemini-2.0-flash")
 
    prompt = f"""
    You are a data expert that converts natural language questions into SQL queries for BigQuery.
    Do not include embedding column in the result.
    The dataset is `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}` with the following columns:
 
    patient_id STRING,
    timestamp TIMESTAMP,
    heart_rate INTEGER,
    bp_systolic INTEGER,
    bp_diastolic INTEGER,
    oxygen_level INTEGER,
    ward STRING,
    risk_score FLOAT,
    status STRING,
    meta STRING,
    patient_notes STRING
 
    IMPORTANT INSTRUCTIONS:
    - If the query asks for "effects", "conditions", "potential effects", or "diagnosis", convert it to retrieve the patient's vital signs data instead.
    - If the query asks for a *report* or *pdf*, ignore those words and still generate a valid SQL query that retrieves the relevant data.
    - For patient-specific queries, use WHERE patient_id = 'PXXX' and ORDER BY timestamp DESC to get the latest readings.
 
    Output only the SQL statement — no explanation, no markdown formatting.
 
    Examples:
    Q: Give me the readings of patient P100 in a pdf report
    A: SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}` WHERE patient_id = 'P100' ORDER BY timestamp DESC LIMIT 10;
    
    Q: what are the potential effects for patient P100
    A: SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}` WHERE patient_id = 'P100' ORDER BY timestamp DESC LIMIT 10;
    
    Q: show me conditions for patient P105
    A: SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}` WHERE patient_id = 'P105' ORDER BY timestamp DESC LIMIT 10;
 
    Query: {query_text}
    """
    resp = model.generate_content(prompt)
    sql = re.sub(r"^```(?:sql)?|```$", "", resp.text, flags=re.MULTILINE).strip()
 
    if not sql or not sql.lower().startswith("select"):
        print("⚠️ Generated SQL is invalid or empty. Response from model:")
        print(resp.text if resp else "❌ No response received from model")
        raise ValueError("Generated SQL is empty or invalid")
 
    print(f"📝 Generated SQL:\n{sql}")
    return sql
 
def execute_bq_sql(sql: str) -> pd.DataFrame:
    print("Executing SQL:\n", sql)
    job = bq_client.query(sql)
    df = job.result().to_dataframe()
    return df

def to_serializable(obj):
    """Recursively convert numpy and other non-serializable types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        # Handle NaN, Infinity, -Infinity
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    # Handle Python float NaN/Inf
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj
 
# ---------- Plotting ----------
def draw_vitals_graph(df: pd.DataFrame, output_dir: str) -> List[str]:
    """Generate vitals graphs and return list of image file paths."""
    img_paths = []
    
    if df.empty:
        print("⚠️ DataFrame is empty, no graphs to generate")
        return img_paths
    
    # Find datetime columns
    time_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    
    if not time_cols:
        print("⚠️ No datetime columns found for time-series graph")
        return img_paths
    
    time_col = time_cols[0]
    print(f"📊 Using time column: {time_col}")
    
    # Sort by time
    df = df.sort_values(by=time_col).copy()
    
    # Find numeric columns to plot (exclude patient_id, status, ward, meta, etc.)
    numeric_cols = []
    skip_cols = [time_col.lower(), "patient_id", "status", "ward", "meta", 
                 "patient_notes", "cosine_similarity", "dot", "norm_t", "norm_q", "embedding"]
    
    for col in df.columns:
        if col.lower() not in skip_cols and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    if not numeric_cols:
        print("⚠️ No numeric columns found to plot")
        return img_paths
    
    print(f"📈 Plotting columns: {numeric_cols}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    for col in numeric_cols:
        try:
            # Drop NaN values for this column
            valid_data = df[[time_col, col]].dropna()
            if not valid_data.empty:
                plt.plot(valid_data[time_col], valid_data[col], marker='o', label=col, linewidth=2)
        except Exception as e:
            print(f"⚠️ Could not plot {col}: {e}")
            continue
    
    plt.title("Patient Vitals Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Readings", fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_file = os.path.join(output_dir, "vitals_chart.png")
    plt.savefig(img_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Graph saved to: {img_file}")
    img_paths.append(img_file)
    
    return img_paths
 
# ---------- PDF ----------
def build_pdf_report_with_graphs(query_text: str, matches: List[Dict[str, Any]], 
                                 output_path: Optional[str] = None,
                                 effects_analysis: Optional[Dict[str, Any]] = None) -> str:
    """Generate a PDF report with patient vitals data, graphs, summary table, and effects analysis."""
    print(f"📄 Building PDF report for query: {query_text}")
    print(f"📊 Number of data rows: {len(matches)}")
    print(f"🏥 Effects analysis provided: {effects_analysis is not None}")
    
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        output_path = tmp.name
        tmp.close()
    
    print(f"💾 PDF will be saved to: {output_path}")
    
    # Convert matches to DataFrame
    df = rows_to_dataframe(matches)
    print(f"📋 DataFrame shape: {df.shape}")
    print(f"📋 DataFrame columns: {list(df.columns)}")
    
    # Create temporary directory for images
    tmpdir = tempfile.mkdtemp(prefix="vitals_imgs_")
    print(f"🖼️ Creating graphs in: {tmpdir}")
    
    # Generate graphs
    images = draw_vitals_graph(df, tmpdir)
    print(f"✅ Generated {len(images)} graph(s)")
 
    # Create PDF
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 40
    
    # ========== PAGE 1: Header + Graph ==========
    y = height - margin
    
    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, y, "CITY HOSPITAL - PATIENT VITALS REPORT")
    y -= 30
    
    # Metadata
    c.setFont("Helvetica", 9)
    c.drawString(margin, y, f"Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    y -= 14
    c.drawString(margin, y, f"Query: {query_text}")
    y -= 30
    
    # Add graph on first page if available
    if images:
        try:
            img = images[0]
            img_reader = ImageReader(img)
            iw, ih = img_reader.getSize()
            aspect = ih / float(iw)
            
            # Calculate image dimensions to fit nicely on page
            available_height = y - margin - 100  # Leave space at bottom
            desired_w = width - 2 * margin
            desired_h = desired_w * aspect
            
            if desired_h > available_height:
                desired_h = available_height
                desired_w = desired_h / aspect
            
            x = (width - desired_w) / 2
            y_img = y - desired_h - 10
            
            c.drawImage(img, x, y_img, width=desired_w, height=desired_h)
            print(f"✅ Added graph to PAGE 1: {img}")
            
            y = y_img - 20  # Update y position after image
        except Exception as e:
            print(f"⚠️ Failed to add image to PAGE 1: {e}")
    
    # ========== PAGE 2: Summary Table ==========
    c.showPage()
    y = height - margin
    
    # Page title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Vitals Summary Statistics")
    y -= 30
 
    if not df.empty:
        # Prepare table data
        table_data = [["Vital Sign", "Mean", "Min", "Max", "Std Dev"]]
        
        # Get numeric columns (exclude metadata columns)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        skip_cols = ["dot", "norm_t", "norm_q", "cosine_similarity"]
        numeric_cols = [col for col in numeric_cols if col not in skip_cols]
        
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            table_data.append([
                col,
                f"{s.mean():.1f}",
                f"{s.min():.1f}",
                f"{s.max():.1f}",
                f"{s.std():.1f}" if len(s) > 1 else "N/A"
            ])
        
        # Add datetime info
        datetime_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if datetime_cols:
            time_col = datetime_cols[0]
            s = df[time_col].dropna()
            if not s.empty:
                table_data.append(["", "", "", "", ""])  # Empty row
                table_data.append(["Time Range", "", "", "", ""])
                table_data.append(["Start", str(s.min()), "", "", ""])
                table_data.append(["End", str(s.max()), "", "", ""])
                duration = s.max() - s.min()
                table_data.append(["Duration", str(duration), "", "", ""])
        
        # Create table
        col_widths = [120, 80, 80, 80, 80]
        table = Table(table_data, colWidths=col_widths)
        
        # Style the table
        table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.black),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F0F0')]),
        ]))
        
        # Draw table
        table_width, table_height = table.wrap(width, height)
        table.drawOn(c, margin, y - table_height)
        
        print(f"✅ Added summary table with {len(table_data)-1} rows")
        
        y -= table_height + 30
        
        # Add additional info
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, "Additional Information")
        y -= 20
        c.setFont("Helvetica", 9)
        
        # Record count
        c.drawString(margin + 10, y, f"Total Records: {len(df)}")
        y -= 14
        
        # Patient IDs
        if 'patient_id' in df.columns:
            patients = df['patient_id'].unique()
            patient_list = ', '.join(str(p) for p in patients[:5])
            if len(patients) > 5:
                patient_list += f" ... (+{len(patients)-5} more)"
            c.drawString(margin + 10, y, f"Patients: {patient_list}")
            y -= 14
        
        # Status distribution
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            status_text = ", ".join([f"{k}: {v}" for k, v in status_counts.items()])
            c.drawString(margin + 10, y, f"Status: {status_text}")
            y -= 14
            
    else:
        c.setFont("Helvetica", 11)
        c.drawString(margin, y, "No data available.")
    
    # ========== PAGE 3: Effects Analysis (Conditions & Potential Effects) ==========
    if effects_analysis and effects_analysis.get("conditions"):
        c.showPage()
        y = height - margin
        
        # Page title
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width / 2, y, "Clinical Conditions & Potential Effects")
        y -= 40
        
        # Vitals analyzed section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Analyzed Vital Signs:")
        y -= 20
        
        c.setFont("Helvetica", 10)
        vitals_analyzed = effects_analysis.get("vitals_analyzed", {})
        for vital_name, vital_value in vitals_analyzed.items():
            display_name = vital_name.replace("_", " ").title()
            c.drawString(margin + 20, y, f"• {display_name}: {vital_value:.1f}")
            y -= 16
        
        y -= 20
        
        # Conditions table
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, f"Identified Conditions ({effects_analysis.get('condition_count', 0)}):")
        y -= 25
        
        # Prepare conditions table data
        conditions_data = [["Condition", "Potential Effects"]]
        
        for condition in effects_analysis.get("conditions", []):
            conditions_data.append([
                condition.get("condition", "Unknown"),
                condition.get("potential_effects", "N/A")
            ])
        
        # Create conditions table
        col_widths = [150, 350]
        conditions_table = Table(conditions_data, colWidths=col_widths)
        
        # Style the conditions table
        conditions_table.setStyle(TableStyle([
            # Header row
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D62828')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 1), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 1), (-1, -1), 10),
            ('RIGHTPADDING', (0, 1), (-1, -1), 10),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#D62828')),
            
            # Alternating row colors
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFEBEE'), colors.HexColor('#FFCDD2')]),
        ]))
        
        # Draw conditions table
        table_width, table_height = conditions_table.wrap(width, height)
        conditions_table.drawOn(c, margin, y - table_height)
        
        print(f"✅ Added effects analysis page with {len(conditions_data)-1} condition(s)")
        
        y -= table_height + 40
        
        # Add disclaimer
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColorRGB(0.3, 0.3, 0.3)
        disclaimer_text = "Note: This analysis is based on the effects master table and should be reviewed by qualified medical professionals."
        c.drawString(margin, y, disclaimer_text)
 
    c.save()
    print(f"✅ PDF report saved successfully: {output_path}")
    return output_path
 
# ---------- Uploader Agent ----------
class UploaderAgent:
    def upload_to_gcs(self, file_path: str, filename: str) -> str:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(filename)
        blob.upload_from_filename(file_path)
        credentials,project = google.auth.default()
        print("------------------------------------------")
        print(client._credentials)
        print("------------------------------------------")
 
        # Generate signed URL safely
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET",
            credentials=client._credentials,
            service_account_email="service-643140588215@gs-project-accounts.iam.gserviceaccount.com",
        )
        return url
 
# ---------- Multi-Agent Workflow ----------
def classify_query_type(query: str) -> str:
    """
    Uses Gemini to decide whether the query should use SQL (analytical)
    or Embedding-based semantic search.
    Returns: "sql" or "embedding"
    """
    model = GenerativeModel("gemini-2.0-flash")
    prompt = f"""
    You are a classifier. Based on the user's query, decide the most suitable retrieval method.
 
    Return exactly one word: "sql" or "embedding".
    Below is the table schema:
    field name	        type
    patient_id	        STRING
    timestamp	        TIMESTAMP
    heart_rate	        INTEGER
    bp_systolic	        INTEGER
    bp_diastolic	    INTEGER
    oxygen_level	    INTEGER
    ward	            STRING
    risk_score	        FLOAT
    status	            STRING
    meta	            STRING
    patient_notes	    STRING

    - Use "sql" for analytical, statistical, or aggregate queries that can be answered with the table columns above (e.g., latest, count, average, min/max, list of patients, show readings, report generation).
    - Use "embedding" for:
      * Descriptive, similarity-based, or record-level queries (e.g., find similar patients, recent patient trends)
      * Questions about "effects", "conditions", "potential effects", "diagnosis" (these require retrieving patient data first)
 
    Example:
    Query: "How many patients were admitted today?" → sql
    Query: "compare the readings between patient1 and patient2" → embedding
    Query: "what are the potential effects for patient P100" → sql (retrieve patient data first)
    Query: "show conditions for this patient" → sql (retrieve patient data first)
 
    Query: {query}
    """
    resp = model.generate_content(prompt)
    result = resp.text.strip().lower()
    if "sql" in result:
        return "sql"
    return "embedding"


class RetrievalAgent:

    def summarize_results(self, query: str, df):
        """
        Uses LLM to create a natural language summary of query results.
        """
        model = GenerativeModel("gemini-2.0-flash")
    
        # Check if query is asking about effects/conditions
        effects_keywords = ["effect", "condition", "diagnosis", "potential effect"]
        is_effects_query = any(keyword in query.lower() for keyword in effects_keywords)
        
        # Convert small DataFrames to markdown for readable context
        if not df.empty and len(df) <= 20:
            table_preview = df.to_markdown(index=False)
        else:
            table_preview = "The table contains many rows."
    
        if is_effects_query:
            # For effects queries, inform that effects analysis will be performed
            prompt = f"""
            The user asked: {query}
            
            The patient's vital signs data has been retrieved successfully.
            
            Provide a brief summary that indicates:
            1. The patient data has been retrieved
            2. Effects analysis will be performed on the vital signs
            3. Mention that clinical conditions will be identified based on the readings
            
            Be concise and professional.
            """
        else:
            # Regular query summary
            prompt = f"""
            You are a data analyst helping summarize query results for hospital management.
            Do not include embedding column in the results.
        
            Here is the original question:
            {query}
        
            Here are the query results:
            {table_preview}
        
            Write a short, clear, human-friendly summary answering the user's question.
            Example:
            Question: How many patients are there?
            Result: 120
            Answer: There are 120 patients currently in the hospital.
        
            Provide only the summary sentence.
            """
    
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, "text") else "Could not summarize results."
    

    def search(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        print(f"🔍 Received query: {query}")
    
        mode = classify_query_type(query)
        print(f"🤖 LLM decided retrieval mode: {mode}")
    
        if mode == "sql":
            try:
                print(f"🧠 Generating SQL for: {query}")
                sql = generate_sql_from_nl(query)
                print(f"📝 Generated SQL:\n{sql}")
                df = execute_bq_sql(sql)
                print(f"✅ Rows fetched: {len(df)}")
                summary = self.summarize_results(query, df)
                print(f"🗣️ Summary:\n{summary}")
                return {
                    "summary": summary,
                    "rows": df.to_dict(orient="records")
                }
            except Exception as e:
                print(f"⚠️ SQL generation/execution failed: {e}")
                return {"summary": "Error occurred during SQL execution.", "rows": []}
        else:
            print("🧮 Using embedding-based similarity search")
            try:
                q_emb = get_query_embedding(query)
                if "patient" in query.lower() and any(ch.isdigit() for ch in query):
                    pid = "".join(ch for ch in query if ch.isdigit())
                    rows = bigquery_cosine_search(q_emb, k=top_k, patient_id=pid)
                elif "all patients" in query.lower() or "every patient" in query.lower():
                    rows = bigquery_cosine_search(q_emb, k=top_k, all_patients=True)
                else:
                    rows = bigquery_cosine_search(q_emb, k=top_k)
    
                df = rows_to_dataframe(rows)
                summary = self.summarize_results(query, df)
                return {
                    "summary": summary,
                    "rows": rows
                }
            except Exception as e:
                print(f"⚠️ Embedding search failed: {e}")
                return {"summary": "Error occurred during embedding search.", "rows": []}
 
class AnalysisAgent:
    def analyze(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        df = rows_to_dataframe(rows)
        stats = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if not s.empty:
                # Calculate statistics, handling NaN/Inf values
                mean_val = s.mean()
                min_val = s.min()
                max_val = s.max()
                std_val = s.std() if len(s) > 1 else 0.0  # std of single value is 0
                
                # Ensure no NaN/Inf values
                stats[col] = {
                    "mean": float(mean_val) if not (np.isnan(mean_val) or np.isinf(mean_val)) else 0.0,
                    "min": float(min_val) if not (np.isnan(min_val) or np.isinf(min_val)) else 0.0,
                    "max": float(max_val) if not (np.isnan(max_val) or np.isinf(max_val)) else 0.0,
                    "std": float(std_val) if not (np.isnan(std_val) or np.isinf(std_val)) else 0.0
                }
        return {"row_count": len(df), "stats": stats}

class EffectsAnalysisAgent:
    """Analyzes patient vitals against the effects_master_table to identify conditions and potential effects."""
    
    def __init__(self):
        self.effects_table = f"{PROJECT_ID}.patient_monitoring.effects_master_table"
    
    def get_conditions_for_vitals(self, heart_rate: Optional[float] = None, 
                                  bp_systolic: Optional[float] = None,
                                  bp_diastolic: Optional[float] = None,
                                  oxygen_level: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Query effects_master_table to find matching conditions based on vital signs.
        Returns a list of conditions with their potential effects.
        """
        conditions = []
        where_clauses = []
        
        # Build WHERE clause for each vital sign
        if heart_rate is not None:
            where_clauses.append(f"""
                (heart_rate_min IS NOT NULL AND heart_rate_max IS NOT NULL 
                 AND {heart_rate} BETWEEN heart_rate_min AND heart_rate_max)
            """)
        
        if bp_systolic is not None:
            where_clauses.append(f"""
                (bp_systolic_min IS NOT NULL AND bp_systolic_max IS NOT NULL 
                 AND {bp_systolic} BETWEEN bp_systolic_min AND bp_systolic_max)
            """)
        
        if bp_diastolic is not None:
            where_clauses.append(f"""
                (bp_diastolic_min IS NOT NULL AND bp_diastolic_max IS NOT NULL 
                 AND {bp_diastolic} BETWEEN bp_diastolic_min AND bp_diastolic_max)
            """)
        
        if oxygen_level is not None:
            where_clauses.append(f"""
                (oxygen_level_min IS NOT NULL AND oxygen_level_max IS NOT NULL 
                 AND {oxygen_level} BETWEEN oxygen_level_min AND oxygen_level_max)
            """)
        
        if not where_clauses:
            return conditions
        
        # Combine with OR to find any matching conditions
        where_sql = " OR ".join(where_clauses)
        
        sql = f"""
        SELECT DISTINCT condition, potential_effects
        FROM `{self.effects_table}`
        WHERE {where_sql}
        ORDER BY condition
        """
        
        try:
            print(f"🔍 Querying effects table with SQL:\n{sql}")
            rows = list(bq_client.query(sql).result())
            
            for row in rows:
                conditions.append({
                    "condition": row["condition"],
                    "potential_effects": row["potential_effects"]
                })
            
            print(f"✅ Found {len(conditions)} matching conditions")
            return conditions
            
        except Exception as e:
            print(f"❌ Error querying effects table: {e}")
            return []
    
    def analyze_patient_vitals(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patient vital signs from rows and return identified conditions.
        """
        if not rows:
            return {"conditions": [], "note": "No data to analyze"}
        
        df = rows_to_dataframe(rows)
        
        if df.empty:
            return {"conditions": [], "note": "Empty dataframe"}
        
        # Get latest or average vitals
        latest_vitals = {}
        
        for col in ["heart_rate", "bp_systolic", "bp_diastolic", "oxygen_level"]:
            if col in df.columns:
                # Use the most recent non-null value
                non_null_values = df[col].dropna()
                if not non_null_values.empty:
                    latest_vitals[col] = float(non_null_values.iloc[-1])
        
        print(f"📊 Latest vitals for analysis: {latest_vitals}")
        
        # Query conditions based on vitals
        conditions = self.get_conditions_for_vitals(
            heart_rate=latest_vitals.get("heart_rate"),
            bp_systolic=latest_vitals.get("bp_systolic"),
            bp_diastolic=latest_vitals.get("bp_diastolic"),
            oxygen_level=latest_vitals.get("oxygen_level")
        )
        
        return {
            "vitals_analyzed": latest_vitals,
            "conditions": conditions,
            "condition_count": len(conditions)
        }
 
class ReportAgent:
    def create_pdf(self, query: str, rows: List[Dict[str, Any]], effects_analysis: Optional[Dict[str, Any]] = None) -> str:
        return build_pdf_report_with_graphs(query, rows, effects_analysis=effects_analysis)
    
class MemoryManager:
    """Stores multi-turn context for conversational reasoning."""
    def __init__(self, max_history: int = 5):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.last_patient_id = None  # Track the most recent patient ID
 
    def add_entry(self, query: str, rows: List[Dict[str, Any]], analysis: Dict[str, Any]):
        # Extract patient_id from rows if available
        patient_id = None
        if rows:
            # Try to get patient_id from first row
            if isinstance(rows, list) and len(rows) > 0:
                if isinstance(rows[0], dict) and "patient_id" in rows[0]:
                    patient_id = rows[0]["patient_id"]
        
        entry = {
            "query": query,
            "timestamp": datetime.utcnow().isoformat(),
            "rows": rows,
            "analysis": analysis,
            "patient_id": patient_id,
        }
        
        # Update last_patient_id if we found one
        if patient_id:
            self.last_patient_id = patient_id
            print(f"💾 Stored patient context: {patient_id}")
        
        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
 
    def get_last(self, n: int = 1) -> Optional[List[Dict[str, Any]]]:
        if not self.history:
            return None
        if n == 1:
            return self.history[-1]["rows"]
        else:
            combined = []
            for entry in self.history[-n:]:
                combined.extend(entry["rows"])
            return combined
 
    def summarize_context(self) -> str:
        """Generate a compact summary of conversation context."""
        if not self.history:
            return ""
        summary = []
        for h in self.history[-self.max_history:]:
            q = h["query"]
            summary.append(f"• {q}")
        return "\n".join(summary)
 
    def get_contextual_query(self, new_query: str) -> str:
        """Merge user's new query with prior context, including patient ID if referring to previous context."""
        if not self.history:
            return new_query
        
        # Keywords that indicate user is referring to previous context
        contextual_keywords = ["that", "this", "these", "those", "their", "the patient", 
                              "same", "previous", "earlier", "effects", "conditions", "readings"]
        
        new_query_lower = new_query.lower()
        is_contextual = any(k in new_query_lower for k in contextual_keywords)
        
        # Check if the query mentions a specific patient ID
        has_explicit_patient = any(f"p{i}" in new_query_lower or f"patient {i}" in new_query_lower 
                                   for i in range(100, 200))
        
        # If contextual and no explicit patient mentioned, inject the last patient ID
        if is_contextual and not has_explicit_patient and self.last_patient_id:
            print(f"🔗 Contextual query detected. Linking to patient: {self.last_patient_id}")
            # Inject patient ID into the query
            enhanced_query = f"{new_query} for patient {self.last_patient_id}"
            context_summary = self.summarize_context()
            return f"{enhanced_query}\nContext from previous queries:\n{context_summary}"
        elif any(k in new_query_lower for k in ["compare", "combine", "both"]):
            context_summary = self.summarize_context()
            return f"{new_query}\nUse the context of previous queries:\n{context_summary}"
        
        return new_query
 
class MainAgent:
    def __init__(self):
        self.retrieval = RetrievalAgent()
        self.analysis = AnalysisAgent()
        self.effects_analysis = EffectsAnalysisAgent()
        self.report = ReportAgent()
        self.uploader = UploaderAgent()
        self.memory = MemoryManager(max_history=5)
 
    def handle_query(self, query: str, top_k: int = TOP_K) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"🔍 MAIN AGENT: Processing query: {query}")
        print(f"{'='*60}\n")
        
        # Store original query for keyword detection
        original_query = query.lower()
        
        contextual_query = self.memory.get_contextual_query(query)
        print(f"📝 Contextual query: {contextual_query}")
        
        # Get search results
        search_result = self.retrieval.search(contextual_query, top_k=top_k)
        print(f"🔎 Search Result Type: {type(search_result)}")
 
        # Support both {'summary': ..., 'rows': [...]} and plain list([...]) results
        if isinstance(search_result, dict):
            summary = search_result.get("summary")
            rows = search_result.get("rows", [])
        elif isinstance(search_result, list):
            summary = None
            rows = search_result
        else:
            # unexpected type: coerce to empty
            summary = None
            rows = []
 
        print(f"📊 Rows Retrieved: {len(rows)} rows")
        
        # Analyze the data
        analysis = self.analysis.analyze(rows)
        print(f"📈 Analysis: {analysis}")
        
        # Analyze conditions and potential effects based on vitals
        effects_analysis = self.effects_analysis.analyze_patient_vitals(rows)
        print(f"🏥 Effects Analysis: {effects_analysis}")
        
        # Save current query into memory
        self.memory.add_entry(contextual_query, rows, analysis)
 
        response = {
            "query": contextual_query,
            "summary": summary,
            "matches": len(rows),
            "analysis": analysis,
            "effects_analysis": effects_analysis,
        }
        
        # Combine results from previous turns if user asks for comparison or combination
        if any(word in original_query for word in ["compare", "combine", "both", "together", "merge"]):
            print("🔄 Combining results from previous queries...")
            past_rows = self.memory.get_last(2)
            if past_rows:
                combined_df = pd.DataFrame(past_rows)
                combined_analysis = self.analysis.analyze(past_rows)
                response["combined_analysis"] = combined_analysis
                response["note"] = "Compared or combined last two queries."
 
        # Generate PDF only when user asks for it
        # Check for various PDF-related keywords in the ORIGINAL query
        pdf_keywords = ["pdf", "report", "download", "export", "save", "file", "document", "graph", "plot", "chart"]
        should_generate_pdf = any(keyword in original_query for keyword in pdf_keywords)
        
        print(f"🔍 PDF keyword check:")
        print(f"   Original query: '{query}'")
        print(f"   Checking keywords: {pdf_keywords}")
        print(f"   Should generate PDF: {should_generate_pdf}")
        
        if should_generate_pdf:
            print(f"\n{'='*60}")
            print(f"📄 PDF GENERATION REQUESTED (detected keyword)")
            print(f"{'='*60}\n")
            
            recent_rows = self.memory.get_last(1)
            print(f"📋 Using {len(recent_rows) if recent_rows else 0} rows for PDF")
            
            if not recent_rows:
                error_msg = "No data found to generate report."
                print(f"❌ {error_msg}")
                response["error"] = error_msg
                return response
            
            try:
                print("🔧 Generating PDF report...")
                pdf_path = self.report.create_pdf(contextual_query, recent_rows, effects_analysis)
                print(f"✅ PDF created at: {pdf_path}")
                
                print("☁️ Uploading to GCS...")
                report_name = f"vitals_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_url = self.uploader.upload_to_gcs(pdf_path, report_name)
                print(f"✅ PDF uploaded to: {pdf_url}")
                
                response["pdf_report_url"] = pdf_url
                response["pdf_local_path"] = pdf_path
                
            except Exception as e:
                error_msg = f"Failed to generate PDF: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                traceback.print_exc()
                response["error"] = error_msg
        
        print(f"\n{'='*60}")
        print(f"✅ Final Response Keys: {list(response.keys())}")
        print(f"{'='*60}\n")
 
        return response
 
# ---------- ADK Agent ----------
main_agent = MainAgent()
 
def adk_tool_handle_query(user_query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Calls the multi-agent pipeline to retrieve, analyze, and report on patient vitals.
    Args:
        user_query: Natural language question from user (e.g. 'Show me Patient 101 vitals').
        top_k: Number of top matches to retrieve from BigQuery.
    Returns:
        Dict containing analysis summary, match count, and optional PDF URL.
    """
    print(f"\n{'#'*60}")
    print(f"🚀 ADK TOOL INVOKED")
    print(f"📝 Query: {user_query}")
    print(f"🔢 Top K: {top_k}")
    print(f"{'#'*60}\n")
    
    try:
        response = main_agent.handle_query(user_query, top_k)
        
        # 🔧 Fix: convert numpy types to serializable Python types
        response = to_serializable(response)
        
        # Build the return object
        result = {
            "summary": response.get("summary", "No summary available."),
            "matches": response.get("matches", 0),
            "analysis": response.get("analysis", {}),
            "effects_analysis": response.get("effects_analysis", {}),
        }
        
        # Add PDF URL if available
        if "pdf_report_url" in response:
            result["pdf_report_url"] = response["pdf_report_url"]
            result["message"] = f"📄 PDF report generated and uploaded successfully!"
        
        # Add error if any
        if "error" in response:
            result["error"] = response["error"]
        
        print(f"\n{'='*60}")
        print(f"✅ ADK TOOL RETURNING: {list(result.keys())}")
        print(f"{'='*60}\n")
        
        return result
        
    except Exception as e:
        error_msg = f"Error in adk_tool_handle_query: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {
            "summary": "An error occurred while processing your query.",
            "error": error_msg
        }
 
root_agent = Agent(
    name="ER_Vitals_Monitoring_MultiAgent",
    model="gemini-2.0-flash",
    description="""👋 Hello! I'm the ER Patient Vital Monitoring Assistant for City Hospital.

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
• 'Compare vitals between patients'

How can I assist you today?""",
    instruction="""
You are an Emergency Room Data Intelligence Assistant integrated with BigQuery.

Your primary task is to invoke the adk_tool_handle_query tool for ALL patient-related queries.

IMPORTANT RULES:
1. **Always pass the user's EXACT query text** to the tool, including keywords like:
   - "pdf", "report", "graph", "chart", "download", "export", "file"
   - "condition", "effects", "diagnosis", "risk", "potential effects"
   - These keywords trigger automatic analysis and PDF generation
   
2. The tool automatically:
   - Retrieves patient data from BigQuery
   - Generates statistical analysis of vital signs
   - **Analyzes conditions and potential effects** based on vital readings
   - Identifies clinical conditions (e.g., Bradycardia, Hypertension, Hypoxemia)
   - Creates time-series graphs of vitals (heart rate, BP, oxygen, etc.)
   - Produces professional PDF reports when requested (with effects analysis)
   - Uploads PDFs to cloud storage and returns download URLs

3. **Never paraphrase or modify the user's query** - pass it exactly as given

4. **When the tool returns effects_analysis**, present it clearly:
   - Show the vitals that were analyzed
   - List each identified condition with its potential effects
   - If no conditions found, state that the vitals are within normal ranges
   - Example format:
     "Based on the analysis of Patient P100's vitals:
     - Heart Rate: 83 bpm
     - Blood Pressure: 149/83 mmHg
     - Oxygen Level: 92%
     
     Identified Conditions:
     1. Stage 1 Hypertension - Potential Effects: Increased cardiovascular risk, may require lifestyle modifications
     2. Mild Hypoxemia - Potential Effects: Reduced oxygen delivery to tissues, may cause fatigue"

5. When the tool returns:
   - summary: Share the natural language summary of results
   - effects_analysis: Present the conditions and potential effects as described above
   - pdf_report_url: Provide the download link to the user

Examples:
- User: "Show me patient P101 vitals"
  → Call tool with: "Show me patient P101 vitals"
  
- User: "What are the potential effects for this patient?"
  → Call tool with: "What are the potential effects for this patient?"
  → Present effects_analysis with conditions and potential effects
  
- User: "what conditions does patient P105 have?"
  → Call tool with: "what conditions does patient P105 have?"
  → Present effects_analysis clearly formatted
""",
    tools=[adk_tool_handle_query]
)

 
class ERMonitoringAgent(BaseAgent):
    def setup(self):
        return root_agent
