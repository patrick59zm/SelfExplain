import pandas as pd
import ast
import textwrap
from rich.console import Console
from rich.table import Table
from rich import box

def pretty_print_inference(
    tsv_path,
    max_samples=None,
    only_wrong=False,
    output_file=None,
    wrap_width=100
):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["true_label", "pred_label", "spans", "links"])

    # Console for display
    live_console = Console(record=False)
    # Console for saving to file
    file_console = Console(record=True)

    printed = 0
    for i, row in df.iterrows():
        is_malformed = False

        # Parse labels
        try:
            true_label = int(row["true_label"])
            pred_label = int(row["pred_label"])
        except Exception:
            true_label = "true_labels"
            pred_label = "predicted_labels"
            is_malformed = True

        correct = true_label == pred_label if not is_malformed else False
        if only_wrong and not is_malformed and correct:
            continue
        if max_samples is not None and printed >= max_samples:
            continue  # Don't count malformed toward max

        table = Table(title=f"Sample #{i+1}", box=box.SQUARE, show_lines=True)
        table.add_column("Field", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Labels
        label_style = "green" if correct else "bold red"
        table.add_row("True Label", str(true_label), style=label_style)
        table.add_row("Predicted Label", str(pred_label), style=label_style)

        # Parse spans
        try:
            span_list = ast.literal_eval(row["spans"])
            original_text = textwrap.fill(span_list[0][0], width=wrap_width)
            formatted_spans = "\n".join(
                [f"[{score:.3f}] {textwrap.fill(span[:300], width=wrap_width)}" for span, score in span_list]
            )
        except Exception as e:
            is_malformed = True
            original_text = "[Error parsing spans]"
            formatted_spans = f"[Error parsing spans] {e}"

        table.add_row("Original Text", original_text)
        table.add_row("Top Spans", formatted_spans)

        # Parse links
        try:
            links = ast.literal_eval(row["links"])
            formatted_links = "\n".join(textwrap.fill(link, width=wrap_width) for link in links)
        except Exception as e:
            is_malformed = True
            formatted_links = f"[Error parsing links] {e}"

        table.add_row("Links", formatted_links)

        # Print logic
        file_console.print(table)
        file_console.rule()

        if not is_malformed:
            if max_samples is None or printed < max_samples:
                live_console.print(table)
                live_console.rule()
                printed += 1

    # Write to file
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(file_console.export_text())
        print(f"[✓] Output written to {output_file}")

# Example usage
if __name__ == "__main__":
    pretty_print_inference(
        tsv_path="phis2000_dev_inference.tsv",
        max_samples=30,
        only_wrong=False,
        output_file="inference_report.md",
        wrap_width=90
    )
