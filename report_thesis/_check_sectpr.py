import docx, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

for i, sec in enumerate(doc.sections):
    print(f'=== Section {i} ===')
    sectPr = sec._sectPr
    frs = sectPr.findall(W + 'footerReference')
    for fr in frs:
        rId = fr.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        ftype = fr.get(W + 'type', 'default')
        print(f'  footerRef type={ftype} rId={rId}')
    
    # Check for firstPage evenPage settings
    for attr in ['titlePg', 'pgNumType']:
        el = sectPr.find(W + attr)
        if el is not None:
            if attr == 'pgNumType':
                print(f'  pgNumType: fmt={el.get(W+"fmt")} start={el.get(W+"start")}')
            else:
                print(f'  {attr}={el.text}')
