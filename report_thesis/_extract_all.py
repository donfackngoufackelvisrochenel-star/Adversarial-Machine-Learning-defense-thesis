import docx, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

doc = docx.Document('report_thesis/MEng_Thesis_Final.docx')

for i, table in enumerate(doc.tables):
    print(f"\n{'='*60}")
    print(f"TABLE {i} ({len(table.rows)} rows x {len(table.columns)} cols)")
    print(f"{'='*60}")
    for row in table.rows:
        cells = [cell.text.strip().replace('\u03b5','eps').replace('\u03b1','alpha') for cell in row.cells]
        print(" | ".join(cells))
