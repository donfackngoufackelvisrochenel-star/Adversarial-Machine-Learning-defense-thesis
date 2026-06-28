import docx, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
rels = doc.part.rels

for i, sec in enumerate(doc.sections):
    print(f'=== Section {i} ===')
    sectPr = sec._sectPr
    frs = sectPr.findall(W + 'footerReference')
    for fr in frs:
        rId = fr.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        ftype = fr.get(W + 'type', 'default')
        rel = rels.get(rId)
        if rel:
            footer_part = rel.target_part
            footer_xml = footer_part._element
            paras = footer_xml.findall('.//' + W + 'p')
            for j, p in enumerate(paras):
                # Check alignment
                jc = p.find('.//' + W + 'jc')
                align = jc.get(W + 'val') if jc is not None else '(no align)'
                texts = p.findall('.//' + W + 't')
                tc = ''.join(t.text or '' for t in texts)[:80]
                print(f'  Footer[{j}]: align={align} text="{tc}"')
