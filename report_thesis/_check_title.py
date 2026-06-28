import docx, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
body = doc.element.body
ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
paras = list(body.iterchildren())

out = []
for i in range(24, 60):
    el = paras[i]
    if el.tag == ns + 'p':
        texts = el.findall('.//' + ns + 't')
        tc = ''.join(t.text or '' for t in texts).strip()
        # Check bold
        run = el.find(ns + 'r')
        bold = ''
        if run is not None:
            rpr = run.find(ns + 'rPr')
            if rpr is not None and rpr.find(ns + 'b') is not None:
                bold = ' [B]'
        # Check font size
        sz = ''
        if run is not None:
            rpr = run.find(ns + 'rPr')
            if rpr is not None:
                sz_el = rpr.find(ns + 'sz')
                if sz_el is not None:
                    sz = f' sz={sz_el.get(ns+"val")}'
        out.append(f'[{i:3d}]{bold}{sz} {tc[:100]}')

with open('C:\\Users\\malware\\Desktop\\final_thesis\\report_thesis\\_title_area.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Done')
