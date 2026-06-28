import docx
from docx.shared import Inches, Pt, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml.ns import qn
from lxml import etree

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
body = doc.element.body
ns = qn('w:')
paras = list(body.iterchildren(ns + 'p'))

# Find TABLE OF CONTENTS and remove the field/instruction paragraphs
# until we hit LIST OF TABLES
toc_idx = None
lot_idx = None
for i, p in enumerate(paras):
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts).strip()
    if tc == 'TABLE OF CONTENTS':
        toc_idx = i
    if tc == 'LIST OF TABLES' and toc_idx is not None:
        lot_idx = i
        break

print(f'TOC at [{toc_idx}], LIST OF TABLES at [{lot_idx}]')

if toc_idx is not None and lot_idx is not None:
    # Remove paragraphs between TOC heading and LIST OF TABLES (the field + instruction)
    to_remove = list(range(toc_idx + 1, lot_idx))
    to_remove.reverse()
    for idx in to_remove:
        body.remove(paras[idx])
    print(f'Removed {len(to_remove)} placeholder paragraphs')

# Re-fetch paragraphs
paras = list(body.iterchildren(ns + 'p'))

# Find the position to insert TOC entries (after TABLE OF CONTENTS)
insert_pos = None
for i, p in enumerate(paras):
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts).strip()
    if tc == 'TABLE OF CONTENTS':
        insert_pos = i
        break

if insert_pos is None:
    print('ERROR: TABLE OF CONTENTS heading not found!')
    exit()

toc_heading_para = paras[insert_pos]

# Define TOC entries: (text, level, page)
# Roman numeral front matter
toc_entries = [
    ("DECLARATION OF ORIGINALITY OF STUDY", 1, "iii"),
    ("ACCEPTANCE OF DISSERTATION", 1, "iv"),
    ("ABSTRACT", 1, "v"),
    ("RESUME", 1, "vi"),
    ("DEDICATION", 1, "vii"),
    ("ACKNOWLEDGMENT", 1, "viii"),
    ("TABLE OF CONTENTS", 1, "ix"),
    ("LIST OF TABLES", 1, "xii"),
    ("LIST OF FIGURES", 1, "xiii"),
    ("LIST OF ABBREVIATIONS", 1, "xiv"),
    ("", 0, ""),  # spacer
    ("CHAPTER 1: INTRODUCTION", 1, "1"),
    ("1.1 Background and Statement of the Problem", 2, "1"),
    ("1.2 Rationale", 2, "4"),
    ("1.3 Research Questions", 2, "5"),
    ("1.3.1 Main Research Question", 3, "5"),
    ("1.3.2 Specific Research Questions", 3, "6"),
    ("1.4 Objectives", 2, "7"),
    ("1.4.1 Overall Objective", 3, "7"),
    ("1.4.2 Specific Objectives", 3, "7"),
    ("1.5 Scope and Delimitations", 2, "8"),
    ("1.6 Structure of the Dissertation", 2, "9"),
    ("", 0, ""),
    ("CHAPTER 2: LITERATURE REVIEW", 1, "10"),
    ("2.1 Overview of Adversarial Machine Learning", 2, "10"),
    ("2.2 Evasion Attacks", 2, "13"),
    ("2.2.1 Fast Gradient Sign Method (FGSM)", 3, "14"),
    ("2.2.2 Projected Gradient Descent (PGD)", 3, "15"),
    ("2.2.3 Comparison of FGSM and PGD", 3, "16"),
    ("2.3 Defense Strategies", 2, "17"),
    ("2.3.1 Adversarial Training", 3, "17"),
    ("2.3.2 Feature Selection Defense", 3, "18"),
    ("2.3.3 Ensemble Defense", 3, "19"),
    ("2.3.4 Feature Squeezing", 3, "20"),
    ("2.4 Evaluation Metrics in Adversarial ML", 2, "21"),
    ("2.5 Related Work on Tabular Adversarial Defenses", 2, "22"),
    ("2.6 Research Gap and Contribution", 2, "24"),
    ("", 0, ""),
    ("CHAPTER 3: MATERIALS AND METHODS", 1, "26"),
    ("3.1 Dataset Description and Preprocessing", 2, "26"),
    ("3.1.1 CIC_IoMT_2024 Dataset", 3, "26"),
    ("3.1.2 Data Loading Pipeline", 3, "28"),
    ("3.1.3 Data Cleaning and Transformation", 3, "29"),
    ("3.1.4 Split Strategy", 3, "30"),
    ("3.1.5 Scaling and Encoding", 3, "31"),
    ("3.2 Model Training and Evaluation", 2, "32"),
    ("3.2.1 Random Forest (RF)", 3, "32"),
    ("3.2.2 XGBoost", 3, "33"),
    ("3.2.3 LightGBM", 3, "34"),
    ("3.2.4 Training Procedure and Hyperparameters", 3, "35"),
    ("3.3 Evasion Attack Implementation", 2, "36"),
    ("3.3.1 FGSM Adaptation for Tree-based Models", 3, "36"),
    ("3.3.2 PGD Adaptation for Tree-based Models", 3, "38"),
    ("3.3.3 Attack Evaluation Protocol", 3, "39"),
    ("3.4 Defense Implementation", 2, "40"),
    ("3.4.1 Adversarial Training", 3, "40"),
    ("3.4.2 Feature Selection Defense", 3, "42"),
    ("3.4.3 Ensemble Defense", 3, "43"),
    ("3.4.4 Feature Squeezing", 3, "44"),
    ("3.5 System Implementation", 2, "45"),
    ("3.5.1 Interactive Dashboard (Streamlit)", 3, "45"),
    ("3.5.2 REST API and WebSocket Server", 3, "47"),
    ("3.5.3 Real-Time Inference Engine", 3, "48"),
    ("3.5.4 Authentication and Security Architecture", 3, "49"),
    ("3.6 Experimental Framework", 2, "50"),
    ("3.6.1 Hardware and Software Configuration", 3, "50"),
    ("3.6.2 Software Libraries", 3, "51"),
    ("3.6.3 Experimental Protocol", 3, "52"),
    ("", 0, ""),
    ("CHAPTER 4: RESULTS AND DISCUSSION", 1, "54"),
    ("4.1 Experimental Results", 2, "54"),
    ("4.1.1 Clean Performance of Base Classifiers", 3, "54"),
    ("4.1.2 Impact of FGSM Evasion Attacks", 3, "57"),
    ("4.1.3 Impact of PGD Evasion Attacks", 3, "59"),
    ("4.1.4 Defense Effectiveness under FGSM", 3, "61"),
    ("4.1.5 Defense Effectiveness under PGD", 3, "64"),
    ("4.2 Discussion", 2, "66"),
    ("4.2.1 Model Selection for IoMT Security", 3, "66"),
    ("4.2.2 Vulnerability of Tree-based Models", 3, "67"),
    ("4.2.3 Defense Trade-offs and Practical Implications", 3, "69"),
    ("4.2.4 Implications for IoMT Security Deployments", 3, "71"),
    ("4.3 Limitations of the Study", 2, "72"),
    ("", 0, ""),
    ("CHAPTER 5: CONCLUSIONS AND RECOMMENDATIONS", 1, "74"),
    ("5.1 Summary of Findings", 2, "74"),
    ("5.2 Conclusions", 2, "75"),
    ("5.2.1 Data Pipeline Implementation", 3, "75"),
    ("5.2.2 Model Performance on Clean Data", 3, "75"),
    ("5.2.3 Attack Vulnerability", 3, "76"),
    ("5.2.4 Defense Effectiveness", 3, "76"),
    ("5.2.5 General Conclusion", 3, "77"),
    ("5.3 Recommendations", 2, "78"),
    ("5.3.1 For Practitioners and System Architects", 3, "78"),
    ("5.3.2 For Future Research", 3, "79"),
    ("", 0, ""),
    ("REFERENCES", 1, "81"),
    ("", 0, ""),
    ("APPENDIX A: SOURCE CODE LISTINGS", 1, "85"),
    ("A.1 Data Loader Module (loader.py)", 2, "85"),
    ("A.2 FGSM Attack Implementation (fgsm.py)", 2, "86"),
    ("A.3 Adversarial Training (adversarial_training.py)", 2, "86"),
    ("A.4 Ensemble Defense (ensemble.py)", 2, "87"),
    ("A.5 Model Trainer (trainer.py)", 2, "87"),
    ("A.6 Evaluation Metrics (metrics.py)", 2, "88"),
    ("A.7 Feature Squeezing (feature_squeezing.py)", 2, "88"),
    ("A.8 SHAP Explainability Module (explainer.py)", 2, "89"),
    ("A.9 Streamlit Dashboard Key Sections (streamlit_app.py)", 2, "89"),
    ("A.10 PGD Attack Implementation (pgd.py)", 2, "90"),
    ("APPENDIX B: ADDITIONAL EXPERIMENTAL RESULTS", 1, "92"),
    ("APPENDIX C: DASHBOARD SCREENSHOTS", 1, "95"),
    ("APPENDIX D: DEPLOYMENT CONFIGURATION", 1, "98"),
    ("APPENDIX E: STATISTICAL ANALYSIS", 1, "100"),
]

# Additional lists that were auto-fields before - keep static too
lot_entries = [
    ("Table 3.1: Model hyperparameters used in experiments", 2, "35"),
    ("Table 4.1: Clean performance of base classifiers on CIC_IoMT_2024 test set", 2, "54"),
    ("Table 4.2: Classifier accuracy under FGSM attack at varying epsilon values", 2, "57"),
    ("Table 4.3: Classifier accuracy under PGD attack at varying epsilon values", 2, "60"),
    ("Table 4.4: Accuracy of defense strategies under FGSM attack", 2, "61"),
    ("Table 4.5: Accuracy of defense strategies under PGD attack", 2, "64"),
    ("Table A1: XGBoost confusion metrics (top 10 classes by false negatives)", 2, "92"),
    ("Table B1: Statistical significance tests for defense comparisons", 2, "93"),
    ("Table B2: Random Forest per-class metrics on clean test data", 2, "93"),
    ("Table B3: LightGBM per-class metrics on clean test data", 2, "94"),
    ("Table B4: Training time and inference latency comparison", 2, "94"),
    ("Table D1: Feature squeezing accuracy under FGSM attack", 2, "97"),
    ("Table E1: 95% confidence intervals for accuracy measurements", 2, "100"),
]

lof_entries = [
    ("Figure 3.1: Correlation heatmap of features in the CIC_IoMT_2024 dataset", 2, "27"),
    ("Figure 3.2: Dashboard view showing trained model metrics", 2, "46"),
    ("Figure 3.3: Dashboard showing attack parameters and results", 2, "47"),
    ("Figure 3.4: Comparison of clean performance metrics across models", 2, "55"),
    ("Figure 4.1: Robustness curves for undefended models under FGSM attack", 2, "58"),
    ("Figure 4.2: Defense comparison under FGSM attack", 2, "62"),
    ("Figure 4.3: Defense comparison under PGD attack", 2, "65"),
    ("Figure 10: Interactive dashboard - main view", 2, "95"),
    ("Figure 11: Best algorithm banner showing metrics", 2, "95"),
    ("Figure 12: Defense comparison view in dashboard", 2, "96"),
    ("Figure 13: SHAP feature importance explanation plots", 2, "96"),
    ("Figure 16: Live inference demo interface", 2, "97"),
    ("Figure 19: Feature squeezing defense accuracy comparison", 2, "98"),
]

# Insert TOC entries after the TABLE OF CONTENTS heading
prev_element = toc_heading_para

# Set page width for tab stop (roughly 6 inches for the right-aligned page number)
page_width_inches = 6.0

for text, level, page in toc_entries:
    if not text and not page:
        # Spacer paragraph
        spacer_xml = (
            '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:r><w:t></w:t></w:r>'
            '</w:p>'
        )
        spacer = etree.fromstring(spacer_xml)
        prev_element.addnext(spacer)
        prev_element = spacer
        continue

    # Build paragraph with right-aligned tab stop + dot leader
    indent = (level - 1) * 0.4  # Level 1=0", Level 2=0.4", Level 3=0.8"

    p_xml = (
        '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:pPr>'
        '<w:ind w:left="' + str(int(indent * 1440)) + '" w:hanging="0"/>'
        '<w:tabs>'
        '<w:tab w:val="right" w:leader="dot" w:pos="' + str(int(page_width_inches * 1440)) + '"/>'
        '</w:tabs>'
        '</w:pPr>'
        '<w:r><w:t xml:space="preserve">' + text + '</w:t></w:r>'
        '<w:r><w:tab/></w:r>'
        '<w:r><w:t>' + page + '</w:t></w:r>'
        '</w:p>'
    )
    toc_entry = etree.fromstring(p_xml)
    prev_element.addnext(toc_entry)
    prev_element = toc_entry

print(f'Inserted {len([e for e in toc_entries if e[0]])} TOC entries')

# Now handle LIST OF TABLES: remove old field, insert static entries
# Find LIST OF TABLES heading
paras = list(body.iterchildren(ns + 'p'))
lot_heading_idx = None
lof_heading_idx = None
lot_next_field_idx = None
lof_next_field_idx = None

for i, p in enumerate(paras):
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts).strip()
    if tc == 'LIST OF TABLES':
        lot_heading_idx = i
    if lot_heading_idx is not None and lot_next_field_idx is None:
        if i > lot_heading_idx:
            lot_next_field_idx = i
            break

paras = list(body.iterchildren(ns + 'p'))
for i, p in enumerate(paras):
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts).strip()
    if tc == 'LIST OF FIGURES':
        lof_heading_idx = i
    if lof_heading_idx is not None and lof_next_field_idx is None:
        if i > lof_heading_idx:
            lof_next_field_idx = i
            break

# Remove old field paragraphs for LoT
if lot_heading_idx is not None and lot_next_field_idx is not None:
    to_remove = list(range(lot_next_field_idx, lof_heading_idx))
    to_remove.reverse()
    for idx in to_remove:
        body.remove(paras[idx])
    print(f'Removed {len(to_remove)} LoT field paragraphs')

    # Insert static LoT entries
    paras = list(body.iterchildren(ns + 'p'))
    for i, p in enumerate(paras):
        texts = p.findall('.//' + ns + 't')
        tc = ''.join(t.text or '' for t in texts).strip()
        if tc == 'LIST OF TABLES':
            insert_after = p
            break

    for text, level, page in lot_entries:
        indent = (level - 1) * 0.4
        p_xml = (
            '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:pPr>'
            '<w:ind w:left="' + str(int(indent * 1440)) + '" w:hanging="0"/>'
            '<w:tabs>'
            '<w:tab w:val="right" w:leader="dot" w:pos="' + str(int(page_width_inches * 1440)) + '"/>'
            '</w:tabs>'
            '</w:pPr>'
            '<w:r><w:t xml:space="preserve">' + text + '</w:t></w:r>'
            '<w:r><w:tab/></w:r>'
            '<w:r><w:t>' + page + '</w:t></w:r>'
            '</w:p>'
        )
        entry = etree.fromstring(p_xml)
        insert_after.addnext(entry)
        insert_after = entry
    print(f'Inserted {len(lot_entries)} List of Tables entries')

# Remove old field paragraphs for LoF
paras = list(body.iterchildren(ns + 'p'))
if lof_heading_idx is not None:
    # Find next heading after LoF (LIST OF ABBREVIATIONS)
    loa_idx = None
    for i, p in enumerate(paras):
        texts = p.findall('.//' + ns + 't')
        tc = ''.join(t.text or '' for t in texts).strip()
        if tc == 'LIST OF ABBREVIATIONS' and lof_heading_idx is not None and i > lof_heading_idx:
            loa_idx = i
            break
    
    if loa_idx is not None:
        # The field paragraph is at lof_heading_idx + 1
        to_remove = list(range(lof_heading_idx + 1, loa_idx))
        to_remove.reverse()
        for idx in to_remove:
            body.remove(paras[idx])
        print(f'Removed {len(to_remove)} LoF field paragraphs')
    
    # Insert static LoF entries
    paras = list(body.iterchildren(ns + 'p'))
    for i, p in enumerate(paras):
        texts = p.findall('.//' + ns + 't')
        tc = ''.join(t.text or '' for t in texts).strip()
        if tc == 'LIST OF FIGURES':
            insert_after = p
            break

    for text, level, page in lof_entries:
        indent = (level - 1) * 0.4
        p_xml = (
            '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:pPr>'
            '<w:ind w:left="' + str(int(indent * 1440)) + '" w:hanging="0"/>'
            '<w:tabs>'
            '<w:tab w:val="right" w:leader="dot" w:pos="' + str(int(page_width_inches * 1440)) + '"/>'
            '</w:tabs>'
            '</w:pPr>'
            '<w:r><w:t xml:space="preserve">' + text + '</w:t></w:r>'
            '<w:r><w:tab/></w:r>'
            '<w:r><w:t>' + page + '</w:t></w:r>'
            '</w:p>'
        )
        entry = etree.fromstring(p_xml)
        insert_after.addnext(entry)
        insert_after = entry
    print(f'Inserted {len(lof_entries)} List of Figures entries')

# Also remove the auto-update setting since we have a static TOC now
settings = doc.settings.element
uf = settings.find(ns + 'updateFields')
if uf is not None:
    settings.remove(uf)
    print('Removed auto-update setting')

doc.save(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
print('Saved with fully static TOC, LoT, and LoF')
