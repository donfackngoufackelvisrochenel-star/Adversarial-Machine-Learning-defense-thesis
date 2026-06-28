import docx, sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
body = doc.element.body
ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
paras = list(body.iterchildren(ns + 'p'))

out_lines = []
out_lines.append(f'Total paragraphs: {len(paras)}')
out_lines.append('')

for i in range(123, min(450, len(paras))):
    p = paras[i]
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts)
    if not tc:
        continue
    has_tab = len(p.findall('.//' + ns + 'tab')) > 0
    ind = p.findall('.//' + ns + 'ind')
    left_indent = ''
    if ind:
        left_indent = ind[0].get(ns + 'left', '0')
    text_show = tc.encode('ascii', 'replace').decode('ascii')[:120]
    marker = ''
    if has_tab:
        marker += ' [TAB]'
    if left_indent and left_indent != '0':
        marker += ' indent=' + left_indent
    out_lines.append(f'[{i:3d}] {text_show}{marker}')

with open(r'C:\Users\malware\Desktop\final_thesis\report_thesis\_toc_verify.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print(f'Wrote {len(out_lines)} lines to _toc_verify.txt')
