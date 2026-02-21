import json
import os
import ollama
from pymongo import MongoClient
from bson import json_util, ObjectId
from state import AgentState
from prompts import TRANSFORM_PROMPT, GENERATE_QUERY_PROMPT

# --- LOAD VARIABLES FROM CONFIG ---
DB_CONNECTION = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_NAME", "FieldOps")

if not DB_NAME:
    raise ValueError("MONGODB_NAME is missing! Please set it in your environment.")

MODEL_NAME = "qwen2.5-coder:14b"
CACHE_FILE = "db_schema_cache.json"

class DBAgentNodes:
    def __init__(self):
        try:
            print(f"Connecting to Database: '{DB_NAME}' at '{DB_CONNECTION}'...")
            self.client = MongoClient(DB_CONNECTION)
            self.db = self.client[DB_NAME]
            
            # Quick check to verify connection
            self.client.admin.command('ping')
            print(f"Connected to MongoDB: {DB_NAME}")
        except Exception as e:
            print(f"Connection Failed: {e}")
            self.db = None

    def _get_schema(self, force_refresh=False):
        if not force_refresh and os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)["schema_text"]
            except Exception:
                pass
        return self._scan_and_cache_db()

    def _scan_and_cache_db(self):
        if self.db is None:
            return "Error: No database connection."
            
        print("Scanning database structure...")
        schema_text = ""
        try:
            for col_name in self.db.list_collection_names():
                if "system." in col_name:
                    continue
                sample = self.db[col_name].find_one()
                if sample:
                    fields = [f"{k} ({type(v).__name__})" for k, v in sample.items() if k != "_id"]
                    schema_text += f"- Collection: '{col_name}' | Fields: {', '.join(fields)}\n"
            
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({"schema_text": schema_text}, f)
        except Exception as e:
            print(f"Schema scan failed: {e}")
            
        return schema_text

    # --- NODE 1: LOAD SCHEMA ---
    def load_schema_node(self, state: AgentState) -> dict:
        # FIX: Force refresh is now True. It will always scan the live DB.
        schema = self._get_schema(force_refresh=True)
        return {"schema_context": schema}

    # --- NODE 2: TRANSFORM QUERY ---
    def transform_query_node(self, state: AgentState) -> dict:
        print(f"Transforming: '{state['raw_input']}'")
        try:
            prompt = TRANSFORM_PROMPT.format(schema_context=state['schema_context'])
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": state['raw_input']}
                ]
            )
            return {"clean_question": response["message"]["content"].strip()}
        except Exception as e:
            return {"error": f"Transformation Failed: {e}"}

    # --- NODE 3: GENERATE JSON ---
    def generate_json_node(self, state: AgentState) -> dict:
        print(f"Generating JSON for: '{state['clean_question']}'")
        try:
            prompt = GENERATE_QUERY_PROMPT.format(
                schema_context=state['schema_context'],
                clean_question=state['clean_question']
            )
            response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
            
            clean_json = response["message"]["content"].replace("```json", "").replace("```", "").strip()
            start, end = clean_json.find("{"), clean_json.rfind("}") + 1
            if start != -1 and end != -1:
                clean_json = clean_json[start:end]
            
            return {"query_json": json.loads(clean_json)}
        except Exception as e:
            return {"error": f"JSON Generation Failed: {e}"}

    # --- NODE 4: EXECUTE & FORMAT ---
    def execute_query_node(self, state: AgentState) -> dict:
        q_data = state.get('query_json')
        if not q_data:
            return {"error": "No query JSON available.", "final_response": "Failed to generate query."}
            
        target_col = q_data.get("collection")
        query_filter = q_data.get("query")
        
        print(f"Target: {target_col} | Filter: {query_filter}")
        
        try:
            # Fetch results
            cursor = self.db[target_col].find(query_filter).limit(5)
            results = list(cursor)
            
            # --- SANITIZE OBJECTID ---
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            # -------------------------

            # Format results for the Supervisor
            if results:
                result_str = json.dumps(results, indent=2, default=str)
                summary = f"Successfully retrieved {len(results)} records from '{target_col}'."
            else:
                result_str = "[]"
                summary = f"No records found in '{target_col}' matching criteria."

            return {
                "query_result": results,
                "final_response": summary + "\nData: " + result_str
            }
        except Exception as e:
            return {"error": str(e), "final_response": f"Database Execution Error: {str(e)}"}

