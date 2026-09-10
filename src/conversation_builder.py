import pandas as pd
from pathlib import Path

INPUT_FILE = "data/raw/twcs.csv"
OUTPUT_FILE = "data/americanair_tweets.csv"

USE_COLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE, usecols=USE_COLS)

print(f"Full dataset: {len(df):,} tweets")

american = df[df["author_id"] == "AmericanAir"].copy()

print(f"AmericanAir tweets: {len(american):,}")

Path("data").mkdir(exist_ok=True)

american.to_csv(OUTPUT_FILE, index=False)

print(f"Saved to: {OUTPUT_FILE}")