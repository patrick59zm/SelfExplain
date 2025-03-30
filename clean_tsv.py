def clean_tsv(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_lines = []
    for i, line in enumerate(lines):
        if i == 0 and "sentence" in line.lower():
            continue  # skip header

        line = line.strip()
        if not line:
            continue

        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue

        sentence, label = parts
        sentence = " ".join(sentence.strip().split())  # remove excess inner spaces
        label = label.strip()

        if label not in {"0", "1"}:
            continue  # skip malformed labels

        cleaned_lines.append(f"{sentence}\t{label}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_lines))


#clean_tsv("data/RoBERTa-SUBJ/train.tsv", "data/RoBERTa-SUBJ/train.tsv")
#clean_tsv("data/RoBERTa-SUBJ/dev.tsv", "data/RoBERTa-SUBJ/dev.tsv")
clean_tsv("data/RoBERTa-SST-2/train.tsv", "data/RoBERTa-SST-2/train.tsv")
clean_tsv("data/RoBERTa-SST-2/dev.tsv", "data/RoBERTa-SST-2/dev.tsv")

