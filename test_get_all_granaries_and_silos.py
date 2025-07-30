import sys
import pandas as pd
from service.scripts.simple_data_retrieval import SimpleDataRetriever, load_config

def test_get_all_granaries_and_silos():
    config = load_config("service/config/streaming_config.json")
    retriever = SimpleDataRetriever(config['database'])
    try:
        df = retriever.get_all_granaries_and_silos()
        print("=== get_all_granaries_and_silos output ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(df)
        # Save to CSV and print output path
        output_path = "granaries_and_silos.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved CSV to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_get_all_granaries_and_silos()
