import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

def clean_text(text):
    """Sanitize input to avoid token alignment issues with benepar."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)              # Normalize all whitespace
    text = re.sub(r"[^\x00-\x7F]+", "", text)     # Remove non-ASCII characters
    return text.strip()

def process_ceas_body_only(csv_path, output_dir, test_size=0.1, dev_size=0.1, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Drop rows with missing body or label
    df = df.dropna(subset=["body", "label"])

    # Clean text and cast label
    df["sentence"] = df["body"].apply(clean_text)
    df["label"] = df["label"].astype(str)

    # Filter out overly long sentences (RoBERTa has max 512 subword tokens)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    df["num_tokens"] = df["sentence"].apply(lambda s: len(tokenizer.tokenize(s)))
    df = df[df["num_tokens"] <= 512].drop(columns=["num_tokens"])

    # Keep only sentence and label
    df = df[["sentence", "label"]]

    # Train/dev/test split
    train_df, temp_df = train_test_split(df, test_size=dev_size + test_size, random_state=seed, stratify=df["label"])
    dev_df, test_df = train_test_split(temp_df, test_size=test_size / (dev_size + test_size), random_state=seed, stratify=temp_df["label"])

    # Save to TSV
    train_df.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False, header=False)
    dev_df.to_csv(os.path.join(output_dir, "dev.tsv"), sep="\t", index=False, header=False)
    test_df.to_csv(os.path.join(output_dir, "test.tsv"), sep="\t", index=False, header=False)

    print(f"✅ Done! Saved {len(train_df)} train / {len(dev_df)} dev / {len(test_df)} test samples to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/phi/CEAS_08.csv")
    parser.add_argument("--output_dir", type=str, default="data/phi")
    parser.add_argument("--dev_size", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.1)
    args = parser.parse_args()

    process_ceas_body_only(args.input_csv, args.output_dir, args.test_size, args.dev_size)
