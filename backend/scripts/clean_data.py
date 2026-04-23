import pandas as pd
import re
import joblib
from pathlib import Path
from tqdm import tqdm

# --- Path Anchoring ---
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "twcs.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed_tickets.joblib"

# --- Domain Logic ---
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

# Instructor Strategy: Expanded Brand Mapping
tech_brands = ['AppleSupport', 'XboxSupport', 'MicrosoftHelps', 'AdobeCare', 'DellCares', 'AskPlayStation']
retail_brands = ['AmazonHelp', 'ArgosHelpers', 'Tesco', 'UPSHelp', 'ChipotleTweets']
travel_brands = ['AmericanAir', 'SouthwestAir', 'British_Airways', 'VirginTrains']
service_brands = ['Uber_Support', 'SpotifyCares', 'AirbnbHelp', 'Delta']

def get_brand_info(text):
    mentions = re.findall(r'@(\w+)', text)
    for m in mentions:
        if m in tech_brands: return m, 'Tech'
        if m in retail_brands: return m, 'Retail'
        if m in travel_brands: return m, 'Travel'
        if m in service_brands: return m, 'Service'
    return 'Unknown', 'Other'

def run_cleaning():
    processed_chunks = []
    # Twitter CSV Date Format: "Tue Oct 31 21:45:10 +0000 2017"
    TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S +0000 %Y"
    
    print(f"🚀 Running Gold Standard Cleaning on {RAW_DATA_PATH}...")
    
    for chunk in tqdm(pd.read_csv(RAW_DATA_PATH, chunksize=100000, low_memory=False), total=30):
        mask = (chunk['inbound'] == True) & (chunk['in_response_to_tweet_id'].isna())
        filtered_chunk = chunk[mask].copy()
        
        if filtered_chunk.empty: continue
            
        # 1. Clean Text & Prune
        filtered_chunk['clean_text'] = filtered_chunk['text'].apply(clean_tweet_text)
        filtered_chunk = filtered_chunk[filtered_chunk['clean_text'].str.len() >= 10]
        
        # 2. Engineering (Metadata)
        filtered_chunk['text_len'] = filtered_chunk['clean_text'].str.len()
        filtered_chunk['priority'] = filtered_chunk['clean_text'].apply(assign_priority)
        
        # 3. Engineering (Temporal - Fixed Format)
        filtered_chunk['created_at'] = pd.to_datetime(filtered_chunk['created_at'], format=TWITTER_DATE_FORMAT, errors='coerce')
        
        # 4. Engineering (Brand/Sector)
        brand_data = filtered_chunk['text'].apply(get_brand_info)
        filtered_chunk['target_brand'] = [x[0] for x in brand_data]
        filtered_chunk['brand_sector'] = [x[1] for x in brand_data]
        
        processed_chunks.append(filtered_chunk)

    full_df = pd.concat(processed_chunks, ignore_index=True)
    joblib.dump(full_df, OUTPUT_PATH)
    print(f"✅ Final Dataset Saved: {len(full_df)} rows with full brand/temporal features.")

if __name__ == "__main__":
    run_cleaning()