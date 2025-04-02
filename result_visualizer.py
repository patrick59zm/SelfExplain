import pandas as pd
import ast
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def pretty_print_inference(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["true_label", "pred_label", "spans", "links"])

    for i, row in df.iterrows():
        table = Table(title=f"Sample #{i+1}", box=box.SQUARE, show_lines=True)

        # Add main info columns
        table.add_column("Field", style="bold cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Labels
        table.add_row("True Label", str(row["true_label"]))
        table.add_row("Predicted Label", str(row["pred_label"]))

        # Parse spans
        try:
            span_list = ast.literal_eval(row["spans"])
            original_text = span_list[0][0]  # Assuming first span = full sentence
            formatted_spans = "\n".join([f"[{score:.3f}] {span[:100]}" for span, score in span_list])
        except Exception as e:
            original_text = "[Error parsing spans]"
            formatted_spans = f"[Error parsing spans] {e}"

        # Show the original sentence
        table.add_row("Original Text", original_text)

        # Show span explanations
        table.add_row("Top Spans", formatted_spans)

        # Parse and show links
        try:
            links = ast.literal_eval(row["links"])
            formatted_links = "\n".join(links)
        except Exception as e:
            formatted_links = f"[Error parsing links] {e}"

        table.add_row("Links", formatted_links)

        # Print the result
        console.print(table)
        console.rule()


# Example usage
if __name__ == "__main__":
    pretty_print_inference("phis2000_dev_inference.tsv")
