from pathlib import Path
from copy import deepcopy
import json, re, hashlib, statistics
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_BREAK
from docx.enum.section import WD_SECTION_START
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

import sys
SUPPLEMENT='--supplement' in sys.argv
R=Path(__file__).resolve().parent
S=json.loads((R.parents[1]/'extensions/cscwd2027_1102/review_v5/numbers.json').read_text(encoding='utf8'))
OUT=R/'output';OUT.mkdir(exist_ok=True)
N=json.loads((R.parents[1]/'analysis/numbers.json').read_text(encoding='utf8'))

XP=R.parents[1]/'extensions/cscwd2027_1102/results/numbers.json'
X=json.loads(XP.read_text(encoding='utf8'))
def cx(condition,comparison='equal_optimizer_average_selected_residual_minus_expanded_selected_FW'):
    return next(v for v in X['contrasts'] if v['test_condition']==condition and v['comparison']==comparison)
def ci_new(v):
    a,b=v['CI95_unadjusted_two_sided_pp'];return f"{v['difference_pp']:.1f} [{a:.1f}, {b:.1f}]"
def sx(role,method,condition):
    return next(v for v in X['test_summary'] if v['role']==role and v['method']==method and v['test_condition']==condition)
def dev(method):return next(v for v in X['development_selection']['expanded_grid'] if v['method']==method and v['learning_rate'] in ['2e-5','3e-4'])

DOCX=OUT/('DexGrasp_CSCWD2027_supplement_1102_v5.docx' if SUPPLEMENT else 'DexGrasp_CSCWD2027_1102_v5.docx')
D=Document(R.parent/'cscwd2027_v4/ieee_template_transitional.docx')
body=D._element.body
for ch in list(body):body.remove(ch)
base=Document(R.parent/'cscwd2027_v4/ieee_template_transitional.docx').sections[5]._sectPr
body.append(deepcopy(base))
for s in D.sections:
    for el in list(s._sectPr):
        if el.tag in [qn('w:headerReference'),qn('w:footerReference')]:s._sectPr.remove(el)
    s.start_type=WD_SECTION_START.NEW_PAGE
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=Inches(.75);s.bottom_margin=Inches(1)
    s.left_margin=Inches(.63);s.right_margin=Inches(.63)
    cols=s._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'1');cols.set(qn('w:space'),'360')
    for c in list(cols):cols.remove(c)
for nm in ['Normal','Body Text','Heading 1','Heading 2','Heading 5','Abstract','Author','Affiliation','Keywords','paper title','references','figure caption','table head','table copy','table col head']:
    st=D.styles[nm];st.font.name='Times New Roman'
    st.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Times New Roman')
D.styles['Normal'].font.size=Pt(10)
D.styles['Body Text'].font.size=Pt(10)
D.styles['Body Text'].paragraph_format.widow_control=True
# IEEE styles contain automatic sample figure/table/reference numbering.
# This manuscript uses explicitly generated numbers; disable the sample lists.
for st in D.styles:
    for el in list(st.element.xpath('./w:pPr/w:numPr')):el.getparent().remove(el)
for nm in ['Heading 1','Heading 2','Heading 5']:
    pp=D.styles[nm].element.get_or_add_pPr()
    for node in list(pp):
        if node.tag==qn('w:numPr'):pp.remove(node)
    D.styles[nm].font.size=Pt(10)
    D.styles[nm].paragraph_format.keep_with_next=True
D.styles['Heading 1'].font.small_caps=True
D.styles['Heading 1'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
D.styles['Heading 2'].font.italic=True
D.styles['Heading 2'].paragraph_format.first_line_indent=Pt(0)
D.styles['Heading 5'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER
D.styles['Abstract'].font.size=Pt(9);D.styles['Abstract'].font.bold=True
D.styles['Abstract'].paragraph_format.space_before=Pt(8)
D.styles['Keywords'].font.size=Pt(9);D.styles['Keywords'].font.italic=True
D.styles['references'].paragraph_format.line_spacing=Pt(9)
D.styles['references'].paragraph_format.left_indent=Inches(.19)
D.styles['references'].paragraph_format.first_line_indent=Inches(-.19)
D.styles['references'].paragraph_format.keep_with_next=False
D.styles['references'].paragraph_format.keep_together=True
D.styles['references'].paragraph_format.widow_control=True

BIB={
162:'Y. Xu et al., “UniDexGrasp: Universal robotic dexterous grasping via learning diverse proposal generation and goal-conditioned policy,” in Proc. IEEE/CVF CVPR, 2023, pp. 4737–4746, doi: 10.1109/CVPR52729.2023.00459.',
157:'W. Wan et al., “UniDexGrasp++: Improving dexterous grasping policy learning via geometry-aware curriculum and iterative generalist-specialist learning,” in Proc. IEEE/CVF ICCV, 2023, pp. 3868–3879, doi: 10.1109/ICCV51070.2023.00360.',
164:'H. Yuan, Z. Huang, Y. Wang, C. Mao, C. Xu, and Z. Lu, “DemoGrasp: Universal dexterous grasping from a single demonstration,” in Proc. ICLR, 2026, arXiv:2509.22149.',
152:'A. Z. Ren et al., “Diffusion policy policy optimization,” in Proc. ICLR, 2025, arXiv:2409.00588.',
131:'F. Ceola, L. Rosasco, and L. Natale, “RESPRECT: Speeding-up multi-fingered grasping with residual reinforcement learning,” IEEE Robot. Autom. Lett., vol. 9, no. 4, pp. 3045–3052, 2024, doi: 10.1109/LRA.2024.3363532.',
140:'Z. Huang, H. Yuan, Y. Fu, and Z. Lu, “Efficient residual learning with mixture-of-experts for universal dexterous grasping,” in Proc. ICLR, 2025, arXiv:2410.02475.',
166:'X. Yuan, T. Mu, S. Tao, Y. Fang, M. Zhang, and H. Su, “Policy Decorator: Model-agnostic online refinement for large policy model,” in Proc. ICLR, 2025, arXiv:2412.13630.',
127:'R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. Bellemare, “Deep reinforcement learning at the edge of the statistical precipice,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 29304–29320.',
136:'T. Eimer, M. Lindauer, and R. Raileanu, “Hyperparameters in reinforcement learning and how to tune them,” in Proc. ICML, PMLR, vol. 202, 2023, pp. 9104–9149.',
151:'A. Patterson, S. Neumann, M. White, and A. White, “Empirical design in reinforcement learning,” J. Mach. Learn. Res., vol. 25, no. 318, pp. 1–63, 2024.',
161:'G.-H. Xu, Y.-L. Wei, D. Zheng, X.-M. Wu, and W.-S. Zheng, “Dexterous Grasp Transformer,” in Proc. IEEE/CVF CVPR, 2024, pp. 17933–17942, doi: 10.1109/CVPR52733.2024.01698.',
165:'H. Yuan, B. Zhou, Y. Fu, and Z. Lu, “Cross-embodiment dexterous grasping with reinforcement learning,” in Proc. ICLR, 2025, pp. 81413–81434.',
156:'A. Wagenmaker et al., “Steering your diffusion policy with latent space reinforcement learning,” in Proc. CoRL, PMLR, vol. 305, 2025, pp. 258–282.',
155:'E. Su, T. Westenbroek, A. Nagabandi, and A. Gupta, “RFS: Reinforcement learning with residual flow steering for dexterous manipulation,” in Proc. ICLR, 2026, arXiv:2602.01789.',
134:'L. Deng et al., “Residual fault adaptation for dexterous in-hand manipulation under runtime joint faults,” arXiv preprint arXiv:2609.17404, 2026.',
159:'M. Wolczyk et al., “Fine-tuning reinforcement learning models is secretly a forgetting mitigation problem,” in Proc. ICML, PMLR, vol. 235, 2024, pp. 53039–53078.',
149:'M. Nakamoto et al., “Cal-QL: Calibrated offline RL pre-training for efficient online fine-tuning,” in Advances in Neural Information Processing Systems, vol. 36, 2023, pp. 62244–62269.',
146:'V. Makoviychuk et al., “Isaac Gym: High performance GPU-based physics simulation for robot learning,” in Proc. NeurIPS Datasets and Benchmarks Track, 2021, arXiv:2108.10470.',
153:'J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.',
154:'Z. Shao et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.',
138:'E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022, arXiv:2106.09685.',
144:'Z. Liu et al., “TAIL: Task-specific adapters for imitation learning with large pretrained models,” in Proc. ICLR, 2024, pp. 16330–16353.'}
BIB.update({
169:'T. Silver, K. Allen, J. Tenenbaum, and L. Kaelbling, “Residual policy learning,” arXiv preprint arXiv:1812.06298, 2018.',
170:'T. Johannink et al., “Residual reinforcement learning for robot control,” in Proc. IEEE ICRA, 2019, pp. 6023–6029, doi: 10.1109/ICRA.2019.8794127.'})
BIB.update({
171:'A. Kumar, A. Raghunathan, R. Jones, T. Ma, and P. Liang, “Fine-tuning can distort pretrained features and underperform out-of-distribution,” in Proc. ICLR, 2022, arXiv:2202.10054.',
172:'M. Wortsman et al., “Robust fine-tuning of zero-shot models,” in Proc. IEEE/CVF CVPR, 2022, pp. 7959–7971.',
173:'J. Kirkpatrick et al., “Overcoming catastrophic forgetting in neural networks,” Proc. Natl. Acad. Sci. USA, vol. 114, no. 13, pp. 3521–3526, 2017, doi: 10.1073/pnas.1611835114.',
174:'A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, “Robots that can adapt like animals,” Nature, vol. 521, pp. 503–507, 2015, doi: 10.1038/nature14422.',
175:'A. Kumar, Z. Fu, D. Pathak, and J. Malik, “RMA: Rapid motor adaptation for legged robots,” in Proc. Robotics: Science and Systems, 2021, doi: 10.15607/RSS.2021.XVII.011.'})
citations=[];content=[];audit={}
def cite(text):
    def repl(m):
        ids=[int(x) for x in m.group(1).split(',')]
        for k in ids:
            assert k in BIB
            if k not in citations:citations.append(k)
        ix=sorted(citations.index(k)+1 for k in ids)
        return ', '.join(f'[{i}]' for i in ix)
    return re.sub(r'\{\{([\d,]+)\}\}',repl,text)
def p(text,style='Body Text'):
    text=cite(text);content.append(text)
    q=D.add_paragraph(text,style);q.paragraph_format.keep_with_next=False
    return q
def h(text):
    q=p(text,'Heading 1');q.paragraph_format.keep_with_next=True
    return q
def sub(text):
    q=p(text,'Heading 2');q.paragraph_format.keep_with_next=True
    return q
def pc(v):return f'{100*v:.1f}'
def interval(v):return f"{100*v['est']:.1f} [{100*v['ci'][0]:.1f}, {100*v['ci'][1]:.1f}]"
def avg(v):return sum(v)/len(v)
def mathrun(txt):
    r=OxmlElement('m:r');pr=OxmlElement('w:rPr')
    fonts=OxmlElement('w:rFonts')
    for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Cambria Math')
    pr.append(fonts);sz=OxmlElement('w:sz');sz.set(qn('w:val'),'20');pr.append(sz);r.append(pr)
    t=OxmlElement('m:t');t.text=txt;r.append(t);return r
def expr(parent,parts):
    for item in parts:
        if isinstance(item,str):parent.append(mathrun(item))
        else:
            typ,base,small=item;node=OxmlElement('m:sSub' if typ=='sub' else 'm:sSup');b=OxmlElement('m:e');expr(b,[base]);node.append(b)
            s=OxmlElement('m:sub' if typ=='sub' else 'm:sup');expr(s,[small]);node.append(s);parent.append(node)
def equation(parts,number):
    # Editable Times-based mathematical runs avoid LibreOffice's Type-3 fallback
    # for OMML operators while keeping genuine Word superscripts/subscripts.
    q=D.add_paragraph();q.alignment=WD_ALIGN_PARAGRAPH.LEFT
    q.paragraph_format.space_before=Pt(4);q.paragraph_format.space_after=Pt(4);q.paragraph_format.first_line_indent=Pt(0)
    q.paragraph_format.keep_together=True
    q.paragraph_format.tab_stops.add_tab_stop(Inches(1.58),WD_TAB_ALIGNMENT.CENTER)
    q.paragraph_format.tab_stops.add_tab_stop(Inches(3.48),WD_TAB_ALIGNMENT.RIGHT)
    q.add_run('\t')
    for item in parts:
        if isinstance(item,str):
            rr=q.add_run(item);rr.font.name='Times New Roman';rr.font.size=Pt(10)
        else:
            typ,base,small=item
            rr=q.add_run(base);rr.font.name='Times New Roman';rr.font.size=Pt(10);rr.italic=True
            rr=q.add_run(small);rr.font.name='Times New Roman';rr.font.size=Pt(10)
            rr.font.subscript=typ=='sub';rr.font.superscript=typ=='sup'
    rr=q.add_run(f'\t({number})');rr.font.name='Times New Roman';rr.font.size=Pt(10)
    content.append('EQUATION '+str(number))
def figure(file,caption):
    q=D.add_paragraph();q.paragraph_format.first_line_indent=Pt(0);q.paragraph_format.space_after=Pt(2);q.paragraph_format.space_before=Pt(4);q.paragraph_format.keep_with_next=True
    img=R/'figures'/file
    if not img.exists():img=R.parent/'cscwd2027_v4/figures'/file
    q.add_run().add_picture(str(img),width=Inches(3.48))
    q=p(caption,'figure caption');q.paragraph_format.first_line_indent=Pt(0)
    q.paragraph_format.keep_together=True
def table(title,headers,rows,widths,note=None):
    q=p(title,'table head');q.paragraph_format.keep_with_next=True;q.paragraph_format.first_line_indent=Pt(0)
    t=D.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for i,s in enumerate(headers):t.rows[0].cells[i].text=s
    for row in rows:
        for c,v in zip(t.add_row().cells,row):c.text=str(v)
    props=t._tbl.tblPr
    mar=OxmlElement('w:tblCellMar')
    for side,value in [('top',35),('bottom',35),('left',25),('right',25)]:
        el=OxmlElement('w:'+side);el.set(qn('w:w'),str(value));el.set(qn('w:type'),'dxa');mar.append(el)
    props.append(mar)
    borders=OxmlElement('w:tblBorders')
    for side in ['top','bottom','left','right','insideH','insideV']:
        el=OxmlElement('w:'+side);el.set(qn('w:val'),'single' if side in ['top','bottom'] else 'nil');el.set(qn('w:sz'),'5');borders.append(el)
    props.append(borders)
    for ri,row in enumerate(t.rows):
        pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
        if ri==0:pr.append(OxmlElement('w:tblHeader'))
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for q in cell.paragraphs:
                q.style=D.styles['table col head' if ri==0 else 'table copy']
                q.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                q.paragraph_format.first_line_indent=Pt(0);q.paragraph_format.left_indent=Pt(0)
                q.paragraph_format.space_before=Pt(0);q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=1
                q.paragraph_format.keep_with_next=ri<len(t.rows)-1 or note is not None
                for r in q.runs:r.font.size=Pt(8);r.bold=ri==0
            if ri==0:
                bp=OxmlElement('w:tcBorders');be=OxmlElement('w:bottom');be.set(qn('w:val'),'single');be.set(qn('w:sz'),'4');bp.append(be);cell._tc.get_or_add_tcPr().append(bp)
    if note:
        q=p(note,'figure caption');q.paragraph_format.space_before=Pt(2);q.paragraph_format.first_line_indent=Pt(0)
        q.paragraph_format.keep_together=True
    audit[title]={'headers':headers,'rows':rows}


def statfmt(v):
    a,b=v['CI95'];return f"{v['mean']:.1f} [{a:.1f}, {b:.1f}]"
def sel(method,condition='LF'):
    return next(v for v in S['same_runtime_selected'] if v['method']==method and v['condition']==condition)
def constrow(i):return S['constant_diagnostics'][i]
