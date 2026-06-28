import docx
doc = docx.Document('report_thesis/MEng_Thesis_Final.docx')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if text:
        print(f'[{i}] {text[:300]}')
