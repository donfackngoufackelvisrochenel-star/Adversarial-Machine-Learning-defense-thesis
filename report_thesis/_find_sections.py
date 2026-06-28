import docx
from docx.oxml.ns import qn

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
body = doc.element.body
ns = qn('w:')

# Find all sectPr elements (section breaks) and what paragraphs they follow
paras = list(body.iterchildren())
for i, el in enumerate(paras):
    if el.tag == ns + 'p':
        texts = el.findall('.//' + ns + 't')
        tc = ''.join(t.text or '' for t in texts).strip()
        # Check if this paragraph contains or is followed by sectPr
        next_sib = el.getnext()
        if next_sib is not None and next_sib.tag == ns + 'sectPr':
            print(f'[P{i}] sectPr AFTER: "{tc[:80]}"')
        # Also check if paragraph itself contains sectPr
        sp = el.find(ns + 'pPr/' + ns + 'sectPr')
        if sp is not None:
            pgNumType = sp.find(ns + 'pgNumType')
            fmt = pgNumType.get(ns + 'fmt') if pgNumType is not None else None
            start = pgNumType.get(ns + 'start') if pgNumType is not None else None
            print(f'[P{i}] INLINE sectPr: "{tc[:80]}"  pgNumType fmt={fmt} start={start}')

# Also check last element for trailing sectPr
last = list(body.iterchildren())[-1]
if last.tag == ns + 'sectPr':
    print(f'LAST element is sectPr')
    pgNumType = last.find(ns + 'pgNumType')
    if pgNumType is not None:
        print(f'  fmt={pgNumType.get(ns+"fmt")} start={pgNumType.get(ns+"start")}')

print(f'\nTotal child elements in body: {len(list(body.iterchildren()))}')
