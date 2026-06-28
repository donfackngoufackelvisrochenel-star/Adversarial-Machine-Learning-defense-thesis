import docx
from docx.shared import Inches, Pt, Emu
from docx.oxml.ns import qn
from lxml import etree

doc = docx.Document(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
body = doc.element.body
ns = qn('w:')

# Build the full TOC, LoT, LoF as a sequential list of paragraphs to insert
# between TABLE OF CONTENTS heading and LIST OF ABBREVIATIONS heading

TOC_ENTRIES = [
    # (text, level, page, type: 'toc', 'lot', 'lof', 'heading', 'spacer')
    # Front matter
    ("DECLARATION OF ORIGINALITY OF STUDY", 1, "iii", "toc"),
    ("ACCEPTANCE OF DISSERTATION", 1, "iv", "toc"),
    ("ABSTRACT", 1, "v", "toc"),
    ("RESUME", 1, "vi", "toc"),
    ("DEDICATION", 1, "vii", "toc"),
    ("ACKNOWLEDGMENT", 1, "viii", "toc"),
    ("TABLE OF CONTENTS", 1, "ix", "toc"),
    ("LIST OF TABLES", 1, "xii", "toc"),
    ("LIST OF FIGURES", 1, "xiii", "toc"),
    ("LIST OF ABBREVIATIONS", 1, "xiv", "toc"),
    ("", 0, "", "spacer"),
    # Chapter 1
    ("CHAPTER 1: INTRODUCTION", 1, "1", "toc"),
    ("1.1 Background and Statement of the Problem", 2, "1", "toc"),
    ("1.2 Rationale", 2, "4", "toc"),
    ("1.3 Research Questions", 2, "5", "toc"),
    ("1.3.1 Main Research Question", 3, "5", "toc"),
    ("1.3.2 Specific Research Questions", 3, "6", "toc"),
    ("1.4 Objectives", 2, "7", "toc"),
    ("1.4.1 Overall Objective", 3, "7", "toc"),
    ("1.4.2 Specific Objectives", 3, "7", "toc"),
    ("1.5 Scope and Delimitations", 2, "8", "toc"),
    ("1.6 Structure of the Dissertation", 2, "9", "toc"),
    ("", 0, "", "spacer"),
    ("CHAPTER 2: LITERATURE REVIEW", 1, "10", "toc"),
    ("2.1 Overview of Adversarial Machine Learning", 2, "10", "toc"),
    ("2.2 Evasion Attacks", 2, "13", "toc"),
    ("2.2.1 Fast Gradient Sign Method (FGSM)", 3, "14", "toc"),
    ("2.2.2 Projected Gradient Descent (PGD)", 3, "15", "toc"),
    ("2.2.3 Comparison of FGSM and PGD", 3, "16", "toc"),
    ("2.3 Defense Strategies", 2, "17", "toc"),
    ("2.3.1 Adversarial Training", 3, "17", "toc"),
    ("2.3.2 Feature Selection Defense", 3, "18", "toc"),
    ("2.3.3 Ensemble Defense", 3, "19", "toc"),
    ("2.3.4 Feature Squeezing", 3, "20", "toc"),
    ("2.4 Evaluation Metrics in Adversarial ML", 2, "21", "toc"),
    ("2.5 Related Work on Tabular Adversarial Defenses", 2, "22", "toc"),
    ("2.6 Research Gap and Contribution", 2, "24", "toc"),
    ("", 0, "", "spacer"),
    ("CHAPTER 3: MATERIALS AND METHODS", 1, "26", "toc"),
    ("3.1 Dataset Description and Preprocessing", 2, "26", "toc"),
    ("3.1.1 CIC_IoMT_2024 Dataset", 3, "26", "toc"),
    ("3.1.2 Data Loading Pipeline", 3, "28", "toc"),
    ("3.1.3 Data Cleaning and Transformation", 3, "29", "toc"),
    ("3.1.4 Split Strategy", 3, "30", "toc"),
    ("3.1.5 Scaling and Encoding", 3, "31", "toc"),
    ("3.2 Model Training and Evaluation", 2, "32", "toc"),
    ("3.2.1 Random Forest (RF)", 3, "32", "toc"),
    ("3.2.2 XGBoost", 3, "33", "toc"),
    ("3.2.3 LightGBM", 3, "34", "toc"),
    ("3.2.4 Training Procedure and Hyperparameters", 3, "35", "toc"),
    ("3.3 Evasion Attack Implementation", 2, "36", "toc"),
    ("3.3.1 FGSM Adaptation for Tree-based Models", 3, "36", "toc"),
    ("3.3.2 PGD Adaptation for Tree-based Models", 3, "38", "toc"),
    ("3.3.3 Attack Evaluation Protocol", 3, "39", "toc"),
    ("3.4 Defense Implementation", 2, "40", "toc"),
    ("3.4.1 Adversarial Training", 3, "40", "toc"),
    ("3.4.2 Feature Selection Defense", 3, "42", "toc"),
    ("3.4.3 Ensemble Defense", 3, "43", "toc"),
    ("3.4.4 Feature Squeezing", 3, "44", "toc"),
    ("3.5 System Implementation", 2, "45", "toc"),
    ("3.5.1 Interactive Dashboard (Streamlit)", 3, "45", "toc"),
    ("3.5.2 REST API and WebSocket Server", 3, "47", "toc"),
    ("3.5.3 Real-Time Inference Engine", 3, "48", "toc"),
    ("3.5.4 Authentication and Security Architecture", 3, "49", "toc"),
    ("3.6 Experimental Framework", 2, "50", "toc"),
    ("3.6.1 Hardware and Software Configuration", 3, "50", "toc"),
    ("3.6.2 Software Libraries", 3, "51", "toc"),
    ("3.6.3 Experimental Protocol", 3, "52", "toc"),
    ("", 0, "", "spacer"),
    ("CHAPTER 4: RESULTS AND DISCUSSION", 1, "54", "toc"),
    ("4.1 Experimental Results", 2, "54", "toc"),
    ("4.1.1 Clean Performance of Base Classifiers", 3, "54", "toc"),
    ("4.1.2 Impact of FGSM Evasion Attacks", 3, "57", "toc"),
    ("4.1.3 Impact of PGD Evasion Attacks", 3, "59", "toc"),
    ("4.1.4 Defense Effectiveness under FGSM", 3, "61", "toc"),
    ("4.1.5 Defense Effectiveness under PGD", 3, "64", "toc"),
    ("4.2 Discussion", 2, "66", "toc"),
    ("4.2.1 Model Selection for IoMT Security", 3, "66", "toc"),
    ("4.2.2 Vulnerability of Tree-based Models", 3, "67", "toc"),
    ("4.2.3 Defense Trade-offs and Practical Implications", 3, "69", "toc"),
    ("4.2.4 Regulatory and Compliance Implications", 3, "71", "toc"),
    ("4.3 Limitations of the Study", 2, "72", "toc"),
    ("", 0, "", "spacer"),
    ("CHAPTER 5: CONCLUSIONS AND RECOMMENDATIONS", 1, "74", "toc"),
    ("5.1 Summary of Findings", 2, "74", "toc"),
    ("5.2 Conclusions", 2, "75", "toc"),
    ("5.2.1 Data Pipeline Implementation", 3, "75", "toc"),
    ("5.2.2 Model Performance on Clean Data", 3, "75", "toc"),
    ("5.2.3 Attack Vulnerability", 3, "76", "toc"),
    ("5.2.4 Defense Effectiveness", 3, "76", "toc"),
    ("5.2.5 General Conclusion", 3, "77", "toc"),
    ("5.3 Recommendations", 2, "78", "toc"),
    ("5.3.1 For Practitioners and System Architects", 3, "78", "toc"),
    ("5.3.2 For Future Research", 3, "79", "toc"),
    ("", 0, "", "spacer"),
    ("REFERENCES", 1, "81", "toc"),
    ("", 0, "", "spacer"),
    ("APPENDIX A: SOURCE CODE LISTINGS", 1, "85", "toc"),
    ("A.1 Data Loader Module (loader.py)", 2, "85", "toc"),
    ("A.2 FGSM Attack Implementation (fgsm.py)", 2, "86", "toc"),
    ("A.3 Adversarial Training (adversarial_training.py)", 2, "86", "toc"),
    ("A.4 Ensemble Defense (ensemble.py)", 2, "87", "toc"),
    ("A.5 Model Trainer (trainer.py)", 2, "87", "toc"),
    ("A.6 Evaluation Metrics (metrics.py)", 2, "88", "toc"),
    ("A.7 Feature Squeezing (feature_squeezing.py)", 2, "88", "toc"),
    ("A.8 SHAP Explainability Module (explainer.py)", 2, "89", "toc"),
    ("A.9 Streamlit Dashboard Key Sections (streamlit_app.py)", 2, "89", "toc"),
    ("A.10 PGD Attack Implementation (pgd.py)", 2, "90", "toc"),
    ("APPENDIX B: ADDITIONAL EXPERIMENTAL RESULTS", 1, "92", "toc"),
    ("B.1 Confusion Matrix for XGBoost (Clean Data)", 2, "92", "toc"),
    ("B.2 Attack Success Rate Comparison", 2, "92", "toc"),
    ("B.3 Per-Class Performance Analysis for Random Forest", 2, "93", "toc"),
    ("B.4 Per-Class Performance Analysis for LightGBM", 2, "93", "toc"),
    ("B.5 Training Time and Inference Latency", 2, "94", "toc"),
    ("B.6 Robustness Comparison Plots", 2, "94", "toc"),
    ("", 0, "", "spacer"),
    ("APPENDIX C: DASHBOARD SCREENSHOTS", 1, "95", "toc"),
    ("C.1 Feature Squeezing Defense Results", 2, "96", "toc"),
    ("", 0, "", "spacer"),
    ("APPENDIX D: DEPLOYMENT CONFIGURATION", 1, "98", "toc"),
    ("D.1 Dockerfile", 2, "98", "toc"),
    ("D.2 Docker Compose", 2, "99", "toc"),
    ("D.3 Requirements", 2, "99", "toc"),
    ("", 0, "", "spacer"),
    ("APPENDIX E: STATISTICAL ANALYSIS", 1, "100", "toc"),
    ("E.1 Confidence Intervals for Accuracy Measurements", 2, "100", "toc"),
    ("E.2 Central Limit Theorem Justification", 2, "101", "toc"),
    ("", 0, "", "spacer"),
    # LIST OF TABLES heading
    ("LIST OF TABLES", 0, "", "heading"),
    ("Table 3.1: Model hyperparameters used in experiments", 2, "35", "lot"),
    ("Table 4.1: Clean performance of base classifiers on CIC_IoMT_2024 test set", 2, "54", "lot"),
    ("Table 4.2: Classifier accuracy under FGSM attack at varying epsilon values", 2, "57", "lot"),
    ("Table 4.3: Classifier accuracy under PGD attack at varying epsilon values", 2, "60", "lot"),
    ("Table 4.4: Accuracy of defense strategies under FGSM attack", 2, "61", "lot"),
    ("Table 4.5: Accuracy of defense strategies under PGD attack", 2, "64", "lot"),
    ("Table A1: XGBoost confusion metrics (top 10 classes by false negatives)", 2, "92", "lot"),
    ("Table B1: Statistical significance tests for defense comparisons", 2, "93", "lot"),
    ("Table B2: Random Forest per-class metrics on clean test data", 2, "93", "lot"),
    ("Table B3: LightGBM per-class metrics on clean test data", 2, "94", "lot"),
    ("Table B4: Training time and inference latency comparison", 2, "94", "lot"),
    ("Table D1: Feature squeezing accuracy under FGSM attack", 2, "97", "lot"),
    ("Table E1: 95% confidence intervals for accuracy measurements", 2, "100", "lot"),
    ("", 0, "", "spacer"),
    # LIST OF FIGURES heading
    ("LIST OF FIGURES", 0, "", "heading"),
    ("Figure 3.1: Correlation heatmap of features in the CIC_IoMT_2024 dataset", 2, "27", "lof"),
    ("Figure 3.2: Dashboard view showing trained model metrics", 2, "46", "lof"),
    ("Figure 3.3: Dashboard showing attack parameters and results", 2, "47", "lof"),
    ("Figure 3.4: Interactive dashboard showing the experimental workflow", 2, "55", "lof"),
    ("Figure 4.1: Comparison of clean performance metrics across models", 2, "58", "lof"),
    ("Figure 4.2: Robustness curves showing accuracy vs. epsilon for undefended models under FGSM attack", 2, "62", "lof"),
    ("Figure 4.3: Defense comparison under FGSM attack", 2, "65", "lof"),
    ("Figure 8: Robustness comparison under FGSM attack (all defenses)", 2, "95", "lof"),
    ("Figure 9: Robustness comparison under PGD attack (all defenses)", 2, "95", "lof"),
    ("Figure 10: Interactive dashboard \u2013 main view showing dataset overview and pipeline audit", 2, "95", "lof"),
    ("Figure 11: Best algorithm banner showing Accuracy, Precision, Recall, and F1 scores", 2, "96", "lof"),
    ("Figure 12: Defense comparison view showing defended vs undefended metrics", 2, "96", "lof"),
    ("Figure 13: SHAP feature importance explanation plots for model interpretability", 2, "97", "lof"),
    ("Figure 16: Live inference demo interface with real-time streaming predictions", 2, "97", "lof"),
    ("Figure 19: Feature squeezing defense accuracy comparison", 2, "98", "lof"),
]

# Helper: create a TOC entry paragraph XML
def make_toc_para(text, level, page, page_width_inches=6.0):
    if level == 0:
        # Heading (LIST OF TABLES / LIST OF FIGURES)
        indent = 0
    else:
        indent = (level - 1) * 0.4  # Level 1=0", Level 2=0.4", Level 3=0.8"
    
    left_twips = str(int(indent * 1440))
    tab_pos = str(int(page_width_inches * 1440))
    
    xml = (
        '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:pPr>'
        '<w:ind w:left="' + left_twips + '" w:hanging="0"/>'
        '<w:tabs>'
        '<w:tab w:val="right" w:leader="dot" w:pos="' + tab_pos + '"/>'
        '</w:tabs>'
        '</w:pPr>'
        '<w:r><w:t xml:space="preserve">' + text + '</w:t></w:r>'
        '<w:r><w:tab/></w:r>'
        '<w:r><w:t>' + page + '</w:t></w:r>'
        '</w:p>'
    )
    return etree.fromstring(xml)

# Step 1: Find TABLE OF CONTENTS heading
paras = list(body.iterchildren(ns + 'p'))
toc_heading = None
for p in paras:
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts).strip()
    if tc == 'TABLE OF CONTENTS':
        toc_heading = p
        break

if toc_heading is None:
    print('ERROR: TABLE OF CONTENTS heading not found')
    exit()

# Step 2: Remove everything between TABLE OF CONTENTS and LIST OF ABBREVIATIONS heading
found_toc = False
to_remove = []
for p in paras:
    texts = p.findall('.//' + ns + 't')
    tc = ''.join(t.text or '' for t in texts).strip()
    if tc == 'TABLE OF CONTENTS':
        found_toc = True
        continue
    if tc == 'LIST OF ABBREVIATIONS' and found_toc:
        break
    if found_toc:
        to_remove.append(p)

print(f'Removing {len(to_remove)} paragraphs between TOC and LIST OF ABBREVIATIONS')
for p in reversed(to_remove):
    body.remove(p)

# Step 3: Insert all new entries after TABLE OF CONTENTS
insert_after = toc_heading
for text, level, page, etype in TOC_ENTRIES:
    if etype == 'spacer':
        # Empty paragraph
        spacer = etree.fromstring(
            '<w:p xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:r><w:t></w:t></w:r></w:p>'
        )
        insert_after.addnext(spacer)
        insert_after = spacer
        continue
    
    para = make_toc_para(text, level, page)
    insert_after.addnext(para)
    insert_after = para

toc_count = sum(1 for e in TOC_ENTRIES if e[3] == 'toc')
lot_count = sum(1 for e in TOC_ENTRIES if e[3] == 'lot')
lof_count = sum(1 for e in TOC_ENTRIES if e[3] == 'lof')
print(f'Inserted: TOC={toc_count}, LoT={lot_count}, LoF={lof_count}')

# Remove auto-update setting (we use static text now)
settings = doc.settings.element
uf = settings.find(ns + 'updateFields')
if uf is not None:
    settings.remove(uf)
    print('Removed auto-update setting')

doc.save(r'C:\Users\malware\Desktop\final_thesis\report_thesis\MEng_Thesis_Final.docx')
print('Saved!')
