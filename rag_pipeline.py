from pathlib import Path

DATA_DIR = Path("data/papers")

def load_papers():
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} papers")
    for p in pdf_files:
        print(p.name)

if __name__ == "__main__":
    load_papers()
