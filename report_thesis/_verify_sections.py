import docx, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
body = doc.element.body
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
paras = list(body.iterchildren())

out = []
out.append(f'Total body children: {len(paras)}')

# Find all sectPr elements
sectPr_locations = []
for i, el in enumerate(paras):
    if el.tag == W + 'p':
        pPr = el.find(W + 'pPr')
        if pPr is not None:
            sp = pPr.find(W + 'sectPr')
            if sp is not None:
                texts = el.findall('.//' + W + 't')
                tc = ''.join(t.text or '' for t in texts).strip()[:80]
                pgNum = sp.find(W + 'pgNumType')
                fmt = pgNum.get(W + 'fmt') if pgNum is not None else None
                start = pgNum.get(W + 'start') if pgNum is not None else None
                titlePg = sp.get(W + 'titlePg')
                out.append(f'SECTPR at P[{i}]: text="{tc}" fmt={fmt} start={start} titlePg={titlePg}')
                # Check footer ref
                fr = sp.find(W + 'footerReference')
                if fr is not None:
                    out.append(f'  footerRef type={fr.get(W+"type")} id={fr.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")}')

# Final sectPr
for child in body.iterchildren():
    if child.tag == W + 'sectPr':
        pgNum = child.find(W + 'pgNumType')
        fmt = pgNum.get(W + 'fmt') if pgNum is not None else None
        start = pgNum.get(W + 'start') if pgNum is not None else None
        out.append(f'FINAL sectPr: fmt={fmt} start={start}')
        fr = child.find(W + 'footerReference')
        if fr is not None:
            out.append(f'  footerRef type={fr.get(W+"type")} id={fr.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")}')

with open('C:\\Users\\malware\\Desktop\\final_thesis\\report_thesis\\_sections_verify.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Done')
print('\n'.join(out))
