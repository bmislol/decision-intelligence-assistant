import pandas as pd
import re
import joblib
from pathlib import Path
from tqdm import tqdm

# --- Absolute Path Anchoring ---
# This finds the directory where clean_data.py lives
SCRIPT_DIR = Path(__file__).resolve().parent 

# Now we navigate relative to the SCRIPT, not the terminal
# Script is in backend/scripts/, so data is up two levels then into data/
BASE_DIR = SCRIPT_DIR.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "twcs.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed_tickets.joblib"

# --- Sanity Check before starting ---
if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(f"❌ Could not find raw data at: {RAW_DATA_PATH.resolve()}")

CHUNK_SIZE = 100000

# --- Modular Functions (from our Notebook) ---
def clean_tweet_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def assign_priority(text):
    high_keywords = ['broken', 'worst', 'error', 'cancel', 'refund', 'waiting', 'emergency', 'help', 'fix', 'cannot']
    low_keywords = ['thanks', 'thank', 'awesome', 'great', 'cool', 'love', 'nice']
    if any(word in text for word in high_keywords): return 2
    if any(word in text for word in low_keywords): return 0
    return 1

# --- Processing Pipeline ---
def run_cleaning():
    processed_chunks = []
    
    print(f"🚀 Starting cleaning process for {RAW_DATA_PATH}...")
    
    # Use tqdm for that "Pro" loading bar
    # total=30 is an estimate for 3M rows / 100k chunks
    for chunk in tqdm(pd.read_csv(RAW_DATA_PATH, chunksize=CHUNK_SIZE, low_memory=False), total=30):
        # 1. Filter for Opening Tickets (Inbound & not a response)
        mask = (chunk['inbound'] == True) & (chunk['in_response_to_tweet_id'].isna())
        filtered_chunk = chunk[mask].copy()
        
        if filtered_chunk.empty:
            continue
            
        # 2. Basic Cleaning
        filtered_chunk['clean_text'] = filtered_chunk['text'].apply(clean_tweet_text)
        filtered_chunk['text_len'] = filtered_chunk['clean_text'].str.len()
        
        # 3. Quality Pruning (Day 2 Logic)
        filtered_chunk = filtered_chunk[filtered_chunk['text_len'] >= 10]
        
        # 4. Engineering (Day 1 Logic)
        filtered_chunk['priority'] = filtered_chunk['clean_text'].apply(assign_priority)
        filtered_chunk['created_at'] = pd.to_datetime(filtered_chunk['created_at'], errors='coerce')
        
        processed_chunks.append(filtered_chunk)

    # Combine everything
    full_df = pd.concat(processed_chunks, ignore_index=True)
    
    # 5. Save as Joblib
    print(f"💾 Saving {len(full_df)} tickets to {OUTPUT_PATH}...")
    joblib.dump(full_df, OUTPUT_PATH)
    print("✅ Data Preparation Complete!")

if __name__ == "__main__":
    run_cleaning()