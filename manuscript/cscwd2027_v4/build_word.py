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

R=Path(__file__).resolve().parent
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

DOCX=OUT/'DexGrasp_CSCWD2027_1102_v4.docx'
D=Document(R/'ieee_template_transitional.docx')
body=D._element.body
for ch in list(body):body.remove(ch)
base=Document(R/'ieee_template_transitional.docx').sections[5]._sectPr
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
    q.add_run().add_picture(str(R/'figures'/file),width=Inches(3.48))
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

title='Recovery–Retention Trade-offs in Dexterous Grasping Policy Adaptation'
q=p(title,'paper title');q.alignment=WD_ALIGN_PARAGRAPH.CENTER
q.paragraph_format.space_after=Pt(10)
authors=D.add_table(rows=1,cols=2);authors.autofit=False
authorlines=[['Haoze Lan','School of Intelligent Manufacturing','(Pen-Tung Sah Institute of Micro-Nano','Science and Technology), Xiamen University','Xiamen 361100, China'],
             ['Yongbin Yuan','Fuzhou University','Fuzhou, China','yuanyongbin@fzu.edu.cn','Corresponding author']]
for cell,lines in zip(authors.rows[0].cells,authorlines):
    cell.width=Inches(3.62);cell.text=''
    for i,line in enumerate(lines):
        q=cell.paragraphs[0] if i==0 else cell.add_paragraph()
        q.style=D.styles['Affiliation'];q.alignment=WD_ALIGN_PARAGRAPH.CENTER
        q.paragraph_format.space_before=Pt(0);q.paragraph_format.space_after=Pt(0);q.paragraph_format.line_spacing=1
        rr=q.add_run(line);rr.font.size=Pt(11);rr.italic=i>0
sec=D.add_section(WD_SECTION_START.CONTINUOUS)
cols=sec._sectPr.find(qn('w:cols'));cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
q=D.paragraphs[-1];q.paragraph_format.space_after=Pt(7);q.paragraph_format.space_before=Pt(0);q.paragraph_format.line_spacing=Pt(1);q.add_run().font.size=Pt(1)

p(f'Abstract—Choosing how to update a pretrained robotic policy requires balancing fault recovery with preservation of nominal behavior. We compare full-weight and residual updates trained with proximal policy optimization (PPO) or group relative policy optimization (GRPO), using one retention-based development rule. Tests cover six Shadow Hand and two Allegro Hand conditions. After extending the full-weight learning-rate grid and evaluating all models on the same server, residual procedures retain a {cx("LF")["difference_pp"]:.1f}-percentage-point advantage at the reference little-finger lock, accompanied by {cx("LF")["nominal_loss_difference_pp"]:.1f} points of additional nominal loss. The large advantage disappears when the finger is locked at 75% of its joint ranges, including after retraining. Replacing a trained residual network with its mean correction achieves {100*statistics.mean(N["const"]["LF"]["const"]["fault"]):.1f}% success without further training. The results connect adaptation choices to achieved retention and fault geometry, and show that a compact constant correction can provide strong reference-lock recovery.','Abstract')
p('Keywords—dexterous grasping, policy adaptation, reinforcement learning, residual learning, capability retention','Keywords')

h('I. INTRODUCTION')
p('Robot cells combine perception, planning and manipulation. When a sensor or actuator changes, designers must decide which policy components to update while preserving nominal operation. Pretrained dexterous policies handle diverse objects {{162,157}}, and DemoGrasp generates an open-loop grasp from one planning decision {{164}}. RL post-training can recover performance after deployment shifts; its value depends on both fault success and the nominal capability retained.')
p('Full-weight fine-tuning changes the pretrained network {{152}}; residual learning adds a correction to a frozen planner {{131,140,166}}. Both alter executed grasps, so their recovery and nominal performance must be measured together. Hyperparameter selection and training variability further shape the resulting operating points {{127,136,151}}.')
p('We ask three questions. RQ1: How do four adaptation procedures trade recovery against retention? RQ2: How do learning-rate selection and lock posture change their relative performance? RQ3: How much correction complexity is needed? Each procedure receives approximately 19,000 grasp attempts and one rate selected jointly on little-finger lock and point-cloud noise. Figure 1 summarizes the comparison.')
p('The study quantifies recovery–retention operating points across eight conditions, identifies a lock-posture boundary that persists after grid refinement, and demonstrates a compact constant-correction baseline. These findings provide evidence for choosing adaptation procedures in deployed robotic systems.')
figure('fig0_pipeline.png','Fig. 1. Comparison workflow. Four update procedures adapt the same pretrained planner. A shared development rule selects their learning rates; evaluation measures both recovery under shifts and retained nominal performance.')

h('II. RELATED WORK')
p('Dexterous grasping. UniDexGrasp and UniDexGrasp++ learn broad grasping behaviors {{162,157}}. Dexterous Grasp Transformer generates configurations from point clouds {{161}}, while CrossDex studies transfer across hands {{165}}. DemoGrasp edits a demonstration through a one-step policy {{164}}, providing the pretrained planner used in this study.')
p('Policy adaptation. Residual policy learning refines existing controllers through an additive learned correction {{169,170}}. RESPRECT and ResDex apply this principle to dexterous grasping {{131,140}}, and Policy Decorator refines pretrained policies online {{166}}. DPPO updates policy weights {{152}}; DSRL and RFS steer frozen generative policies through latent coordinates {{156,155}}. Other compact updates include LoRA {{138}} and task-specific adapters {{144}}. Residual Fault Adaptation uses recurrent control and execution history for runtime joint faults in in-hand manipulation {{134}}. Here, adaptation operates through the released one-step planner interface.')
p('Retention and evaluation. Forgetting during RL fine-tuning {{159}} and unstable offline-to-online transitions {{149}} motivate explicit retention measurements. Tuning and independent training runs are also central to reliable comparisons {{127,136,151}}. Our shared selection rule connects these issues to fault recovery and the complexity of the learned correction.')

h('III. ADAPTATION DESIGN AND PROTOCOL')
sub('A. Planner and deployment conditions')
p('We use released Shadow and Allegro DemoGrasp checkpoints in Isaac Gym {{164,146}}. An observation s contains the initial end-effector pose, object pose and point cloud. The planner edits the trajectory and grasp pose, executes open loop, and receives a binary lift outcome. The Shadow action has 24 dimensions: three wrist translations, three wrist rotations and 18 hand-pose edits. Its pretrained Gaussian policy samples a pre-squashing action u and executes a = tanh(u):')
equation(['u = ',('sub','m','0'),'(s) + ',('sub','σ','0'),' ⊙ ε,     ε ∼ N(0, I).'],1)
p('The pretrained mean and sampling scale are m₀ and σ₀; ⊙ denotes elementwise multiplication. All procedures use the released observation interface, which excludes joint state and fault labels. Reference locks hold the joint targets from the first demonstration frame.')
p('The six Shadow conditions are little-finger lock (LF), thumb lock (TH), 1 cm Gaussian point-cloud noise per coordinate (PCL), a vertical calibration offset U(−3, 3) cm (ZR), a fixed 2 cm offset with PCL (Z2+PCL), and TH+PCL. Allegro uses TH and the fixed 2 cm offset (Z2). Evaluation-only tests combine LF+PCL and TH+ZR or change lock posture. Training uses 768 objects and binary lift rewards.')
p('A grasp fails if a palm or fingertip keypoint crosses below the tabletop. The primary simulation retains the released collision filter, which disables physical robot–table collisions in half the parallel environments. The geometric failure check remains active in all environments. A separate evaluation enables all robot–table collisions and adds a ground plane.')

sub('B. Four update procedures')
p('Full-weight PPO {{153}} updates the planner and sampling scale with a learned value baseline. Full-weight GRPO {{154}} updates the backbone and mean head, fixes its sampling scale to max(σ₀, 0.05), and samples eight actions per state. GRPO normalizes rewards within each group; groups with identical rewards contribute no gradient. Residual procedures freeze the planner and sample')
equation(['u = ',('sub','m','0'),'(s) + ',('sub','σ','0'),' ⊙ [',('sub','δ','ϕ'),'(s) + ',('sup','e','ρ'),' ⊙ ε].'],2)
p('The correction network δϕ uses planner features, layer normalization and two 256-unit hidden layers. Its output layer and log-scale vector ρ start at zero, reproducing the pretrained policy. Residual PPO adds a value network; residual GRPO uses the same 64-state, eight-action group construction as full-weight GRPO. The four procedures retain their specified scale treatment and update schedules (Table I).')
table('TABLE I. UPDATE CONFIGURATIONS',
      ['Procedure','Sampling scale','Epochs','Attempts'],
      [['Full-weight PPO','Learned','5','19,712'],['Full-weight GRPO','max(σ₀, 0.05)','2','19,456'],['Residual PPO','σ₀ ⊙ exp(ρ)','5','18,944'],['Residual GRPO','σ₀ ⊙ exp(ρ)','5','18,944']],
      [1.10,1.01,.58,.79],
      'Full-weight PPO follows the released three-phase schedule; the others use 12 phases. Full-weight PPO/GRPO use 4.1%/2.7% more interactions than residual procedures. Each procedure evaluates its final checkpoint.')

sub('C. Retention-constrained selection')
p('The development set contains 64 objects, disjoint from training and test objects. Each scheme uses one learning rate shared by LF and PCL. Full-weight candidates are {10⁻⁵, 3×10⁻⁵, 10⁻⁴}; residual candidates are {10⁻⁴, 3×10⁻⁴, 10⁻³, 3×10⁻³, 10⁻²}. A rate is feasible if its mean nominal loss is at most five percentage points (pp) in both conditions. Among feasible rates, the highest average fault success wins; ties favor smaller worst nominal loss and then a smaller rate. Selection uses unrounded values from 512 fault and 2,048 nominal attempts per model.')
p('Each candidate starts with three independent training runs. A one-time rule adds two runs in both development conditions for candidates initially within one point of the feasibility boundary; selection then uses all five. The original grid selects 10⁻⁵ for both full-weight schemes, 3×10⁻³ for residual PPO and 10⁻³ for residual GRPO. These rates transfer unchanged to separately trained models for the remaining conditions. This is transfer of a training configuration, including to Allegro, rather than transfer of a learned policy across hands.')

sub('D. Evaluation and statistics')
p('Evaluation executes the policy mean with seed 4242. Test objects comprise 141 seen-category instances (1,128 attempts) and 78 unseen-category instances represented in 100 slots (800 attempts). Each run receives the average of its seen-category and unseen-category success rates. Nominal loss is the frozen planner’s nominal score minus the adapted score; recovery is the gain over the frozen planner under the same shift.')
p('The protocol fixed configurations and directional tests before the final evaluation, after exploratory use of the test objects. LF/PCL evaluations reuse selection runs. Statistical units are independent training runs (three to five per group). We report two-sided unadjusted 95% Welch–Satterthwaite intervals; the original nine directional tests use one-sided p-values with Holm correction (Table IV). Later analyses are descriptive.')
p('The residual-minus-full-weight contrast gives equal weight to the PPO and GRPO contrasts. Optimizer contrasts similarly average over the two update types. Seen and unseen scores are weighted within each run; cross-procedure inference uses independent runs. The selected residual PPO LF/PCL groups contain five runs, and the other primary groups contain three. Data and reconstruction scripts are available in the repository.¹')

h('IV. RECOVERY, RETENTION AND FAULT GEOMETRY')
sub('A. Recovery across deployment shifts')
p('Under the original protocol (Table II), the strongest separation occurs at reference LF: the frozen planner succeeds on 16.9% of attempts, full-weight schemes reach about 32%, and residual schemes reach 64–65%. The averaged recovery contrast is '+interval(N['main']['LF']['struct_fault'])+' pp, accompanied by '+pc(-N['main']['LF']['struct_nom']['est'])+' pp more nominal loss. This operating point offers a large recovery gain at a measurable retention cost.')
rows=[]
for key,label in [('LF','LF'),('PCL','PCL'),('TH','TH'),('ZR','ZR'),('BZ2PN','Z2+PCL'),('THPN','TH+PCL')]:
    x=N['main'][key];row=[label,pc(x['frozen_fault'])]
    for method in ['full_ppo','full_grpo','res_ppo','res_grpo']:
        z=x[method];row.append(pc(z['fault_mean'])+'\n('+pc(N['frozen']['nom']-z['nom_mean'])+')')
    rows.append(row)
for key,label in [('TH','A: TH'),('Z2','A: Z2')]:
    x=N['allegro'][key];row=[label,pc(x['frozen_fault'])]
    for method in ['full_ppo','full_grpo','res_ppo','res_grpo']:
        z=x['groups'][method];row.append(pc(avg(z['fault']))+'\n('+pc(x['frozen_nom']-avg(z['nom']))+')')
    rows.append(row)
table('TABLE II. ORIGINAL SUCCESS (%) AND NOMINAL LOSS (pp)',
      ['Shift','Frozen','Full\nPPO','Full\nGRPO','Residual\nPPO','Residual\nGRPO'],rows,[.48,.43,.64,.64,.64,.65],
      'Parentheses give nominal loss relative to the frozen planner. A: Allegro; all other rows use Shadow. Full: full-weight updates. Frozen nominal success is 91.0% for Shadow and 86.8% for Allegro. Three runs per cell, except residual PPO LF/PCL (five runs).')
p('Full-weight means are slightly higher in the other seven conditions. Optimizer contrasts range from −0.3 to +1.8 pp, much smaller than the reference-lock procedure contrast. On Allegro, six of eight adapted method–condition means exceed five points of nominal loss. Configuration transfer therefore requires checking retention in the target deployment condition.')
p('Figure 2 illustrates three LF outcomes: recovery by residual and scaled-shift policies, a shared failure, and a shared success. The objects and outcomes are taken from the archived simulation visualization; aggregate comparisons use Table II.')
figure('fig3_sim_examples.png','Fig. 2. Isaac Gym simulation renders under little-finger lock. Three purposefully selected objects, not a random sample; final frames from the first training run and first seen-category episode. S/F denotes the recorded success/failure flag. The scaled shift is the 24-parameter correction in Section V-B. Two of nine visualization reruns differed from archived outcomes; all displayed objects exclude these discrepancies.')

sub('B. Selection changes the operating point')
p('Figure 3 shows the recovery–retention operating points and individual training runs. At similar test nominal losses, full-weight PPO at 3×10⁻⁵ reaches 50.2% success with 5.1 pp loss, while residual PPO reaches 65.2% with 5.4 pp loss. Their 15.0 pp separation provides a second view of the recovery gain alongside the development-selected comparison.')
figure('fig1_tradeoff.png','Fig. 3. Reference-LF recovery versus nominal loss. Lines connect archived candidate means; small translucent markers show independent training runs. Stars mark development-selected rates. The horizontal dotted line is the frozen planner. The plot covers 14 of the 16 original candidate configurations; both residual 10⁻² candidates are omitted.')
p('Selection changes the size of the advantage. Full-weight PPO at 3×10⁻⁵ misses the shared constraint by 0.20 pp on development PCL; full-weight GRPO at that rate loses 8.45 pp on development LF. LF-only selection gives a 23.5 pp averaged contrast, and a shared ten-point limit gives 18.0 pp. Section IV-D adds an intermediate rate to both full-weight grids.')
p('With ideal fault routing, nominal episodes use the frozen planner and fault-model selection drops the retention limit. Residual procedures retain a '+interval(N['tests']['H5'])+' pp advantage (Holm p = '+f"{N['tests']['H5']['p_holm']:.3f}"+'), showing that routing does not remove the recovery difference.')

sub('C. The advantage depends on lock posture')
p('Alternative lock postures place each controlled little-finger joint at 25% or 75% of its own range. Reference-trained models retain a '+pc(N['boundary']['LF25']['fault']['est'])+' pp residual advantage at 25%; at 75%, the contrast is '+interval(N['boundary']['LF75_eval']['fault'])+' pp (Fig. 4). Frozen success is 16.9%, 19.2%, 16.0% and 16.6% for reference, 25%, 75% and two-finger locks. The same nominal evaluation applies across test-time postures.')
figure('fig2_posture.png','Fig. 4. Residual-minus-full-weight success and unadjusted two-sided 95% intervals. Circles: original selection; squares: expanded full-weight selection with the original residual choices, reevaluated on the same server. The first three rows reuse reference-trained models. The lower rows use separate models; two-finger groups have five runs, including the initial three. Frozen values at right use the original protocol.')
p('A direct within-run comparison gives a 32.9 [27.6, 38.3] pp reduction in the procedure contrast from reference to 75%. Retraining at 75% brings all four procedures to roughly 41–43% success, with an averaged contrast of '+interval(N['boundary']['LF75_retrain']['fault'])+' pp. With ring and little fingers locked together, the residual advantage is '+interval(N['boundary']['RFLF']['fault'])+' pp. Lock posture changes which correction is effective.')

sub('D. Expanded-grid and matched-setting results')
p(f'Adding 2×10⁻⁵ to the full-weight grids changes the selected PPO rate to 2×10⁻⁵; GRPO retains 10⁻⁵. The extension completes {X["new_training_chains"]} training runs, including the prescribed additions in both full-weight GRPO conditions. All comparisons in Table III use the same evaluation server. Full-weight configurations are reselected on development data; residual configurations retain their original selected rates.')
extensionrows=[]
for label,role,method in [('FW-PPO','expanded_selected_fullweight','fwppo'),('FW-GRPO','expanded_selected_fullweight','fwgrpo'),('Residual PPO','original_selected_residual','res_ppo'),('Residual GRPO','original_selected_residual','res_grpo'),('Matched FW-GRPO','matched_fullweight_reference','fwgrpo_matched_reference'),('Matched residual','matched_residual','matchedres')]:
    a,b=sx(role,method,'LF'),sx(role,method,'pn10')
    rate={'2e-5':'2×10⁻⁵','1e-5':'10⁻⁵','3e-3':'3×10⁻³','1e-3':'10⁻³','3e-4':'3×10⁻⁴'}[a['learning_rate']]
    extensionrows.append([label,f'{rate} / {a["n"]}',f'{a["success_mean_pct"]:.1f} ({a["nominal_loss_mean_pp"]:.1f})',f'{b["success_mean_pct"]:.1f} ({b["nominal_loss_mean_pp"]:.1f})'])
table('TABLE III. SAME-RUNTIME EXTENSION RESULTS', ['Procedure','Rate / n','LF','PCL'],extensionrows,[1.22,.78,.74,.74],
      f'Success (%) and nominal loss (pp, parentheses). Frozen nominal success: {X["frozen_test_success_pct"]["nom"]:.1f}%. Matched FW-GRPO is an independently trained reference. The final two rows use fixed sampling scale and two update epochs. n is training runs per condition.')
p(f'The reference-lock advantage remains {ci_new(cx("LF"))} pp after grid refinement, with {cx("LF")["nominal_loss_difference_pp"]:.1f} pp additional nominal loss. The PPO contrast is {ci_new(next(v for v in X["contrasts"] if v["comparison"]=="selected_residual_minus_expanded_selected_FW" and v["optimizer"]=="ppo" and v["test_condition"]=="LF"))} pp. At 25% and 75%, the averaged contrasts are {ci_new(cx("LF25"))} and {ci_new(cx("LF75"))} pp, respectively. Under point-cloud noise, full-weight updates lead by {-cx("pn10")["difference_pp"]:.1f} pp: the residual-minus-full-weight contrast is {ci_new(cx("pn10"))} pp. The extension preserves both the reference-lock recovery advantage and its posture boundary.')
p(f'Matching residual GRPO to the full-weight sampling scale max(σ₀, 0.05), frozen ρ and two epochs retains strong recovery: the added 3×10⁻⁴ rate achieves {sx("matched_residual","matchedres","LF")["success_mean_pct"]:.1f}% LF success. Its development LF loss is {dev("matchedres")["LF_loss_pp"]:.1f} pp, above the five-point limit. Full-weight GRPO at 2×10⁻⁵ also fails the shared rule, with {dev("fwgrpo")["PCL_loss_pp"]:.1f} pp PCL development loss. The selected operating point therefore depends on retention as well as recovery.')
p(f'Reevaluating the same 40 archived checkpoints on the second GPU server changes mean development nominal loss by {X["environment_drift"]["summary"]["paired_loss_change_pp"]["mean"]:.2f} pp. Original-grid feasibility is unchanged, so PPO’s new choice comes from the added rate. All new recovery and nominal-loss contrasts use the reevaluated models and frozen reference.')
p('Two further checks reinforce the operating-point interpretation. A KL-regularized full-weight GRPO baseline reaches '+pc(N['kl']['compare']['LF']['means']['kl']['fault'])+'% LF success at '+pc(N['frozen']['nom']-N['kl']['compare']['LF']['means']['kl']['nom'])+' pp loss. Its rate 3×10⁻⁵ and coefficient 0.00528 are selected jointly on LF/PCL; coefficients {0.1, 1}/K̄ are calibrated from six residual-GRPO mean-shift diagnostics. Enabling all robot–table collisions and adding a ground plane changes the original LF procedure contrast by 0.2 [−0.9, 1.4] pp.')

testspec=[('H1','Reference LF','H1'),('H2','LF versus PCL','H2'),('H3a','25% lock','H3a'),('H3b','75% lock','H3b'),('H4a','LF to LF+PCL','H4_LFpn_srcLF'),('H4b','PCL to LF+PCL','H4_LFpn_srcPCL'),('H4c','TH to TH+ZR','H4_THzr_srcTH'),('H4d','ZR to TH+ZR','H4_THzr_srcZR'),('H5','Ideal routing','H5')]
testrows=[]
for label,meaning,key in testspec:
    x=N['tests'][key];pvalue='<0.001' if x['p_holm']<.001 else f"{x['p_holm']:.3f}"
    if label in ['H4a','H4b','H4c','H4d']:
        entry={'H4a':('LF → LF+PCL','LFpn_srcLF'),'H4b':('PCL → LF+PCL','LFpn_srcPCL'),'H4c':('TH → TH+ZR','THzr_srcTH'),'H4d':('ZR → TH+ZR','THzr_srcZR')}[label]
        gains=N['extra']['H4'][entry[1]]['avg']
        meaning=entry[0]+f'\n{100*gains["G_A"]:.1f} → {100*gains["G_AB"]:.1f}'
    testrows.append([label,meaning,interval(x),pvalue])
table('TABLE IV. ORIGINAL NINE DIRECTIONAL TESTS',
      ['Test','Setting / gain','Effect [95% CI]','Holm p'],testrows,[.35,1.08,1.53,.52],
      'Effects and unadjusted two-sided 95% Welch intervals in pp; decisions use one-sided p-values with Holm correction. H4 reuses single-shift models on combined shifts: setting labels show mean recovery gain over frozen before → after combination, and effects measure its reduction. H5 uses ideal fault routing.')


h('V. COMPACT CORRECTIONS')
sub('A. Constant corrections recover reference-lock performance')
p('A common offset recovers reference-lock performance without further training. Replacing the residual GRPO network’s mean output with its average over nominal training-object states yields 66.7% success at 4.2 pp nominal loss, compared with 64.2% at 4.9 pp for the source network (Table V). A directly trained constant plus scale reaches 67.1% at 4.1 pp loss. The frozen planner continues to provide state-dependent grasp proposals.')
lf=N['const']['LF'];bias=N['supp_1008']['bias_control']['stats']['conditions']
direct=N['cst']['grid_test']['LF'][N['cst']['select']['LF']['selected_lr']]
constrows=[['Residual GRPO','State-dependent',pc(avg(lf['full']['fault'])),pc(lf['Y0_nom']-avg(lf['full']['nom']))],
           ['Post-training constant','Constant',pc(avg(lf['const']['fault'])),pc(lf['Y0_nom']-avg(lf['const']['nom']))],
           ['Direct constant + scale','48',pc(direct['fault_mean']),pc(direct['nom_loss'])]]
for label,key in [('Scaled output shift','sigma|3e-2'),('Raw output shift','raw|3e-3')]:
    constrows.append([label,'24',pc(avg(bias['LF|LF']['runs'][key])),pc(N['frozen']['nom']-avg(bias['LF|nom']['runs'][key]))])
table('TABLE V. ORIGINAL CORRECTION DIAGNOSTICS',
      ['Correction','Form/parameters','Success','Loss'],constrows,[1.31,1.03,.57,.57],
      'Success in %, nominal loss in pp; n = 3 throughout. The first two rows share trained models. Direct constant + scale selects on LF; the 24-parameter arms select jointly on LF/PCL and fix sampling scale.')
p('Correction complexity matters more under the other lock configurations. At the retrained 75% posture, replacing the network with its mean changes success from 41.1% to 36.7%. Under two-finger lock, the network, post-training constant and directly trained constant achieve 36.5%, 32.5% and 26.8%. The compact correction is most effective at reference LF.')

sub('B. Output coordinates and correction structure')
p('A constant mean shift can be represented as an output-bias change. For a frozen linear mean head m₀(s) = Wh₀(s) + b, define d = σ₀ ⊙ c. Then')
equation([('sub','m','0'),'(s) + d = W',('sub','h','0'),'(s) + (b + d).'],3)
p('Both 24-parameter arms represent the same constant shifts. Joint LF/PCL selection gives 67.6% success at '+constrows[3][3]+' pp loss in scaled coordinates and 27.1% at 3.3 pp in raw coordinates. The raw arm also reaches 68.7% at 8.5 pp loss at a larger rate. Their selection-dependent operating points reflect optimization coordinates, rather than different representational capacity; the scale-matched rate comparisons have intervals spanning zero.')
p(f'Checkpoint outputs show a concentrated correction. Across 2,048 nominal reference states, residual GRPO allocates 87% of absolute mean movement to the 12 largest pretrained-scale dimensions, versus 62% for full-weight GRPO. State variation accounts for 25% of its squared movement. After tanh and joint limits, RFJ3’s mean target edit is 1.29 rad versus 0.28 rad. At the retrained 75% posture, residual GRPO instead has a {N["supp_1008"]["output_analysis"]["extra_groups"]["rgLF75"]["mean"]["cos_ref"]:.2f} cosine with the reference correction direction and a {N["supp_1008"]["output_analysis"]["extra_groups"]["rgLF75"]["mean"]["dq_rfj3"]:.2f} rad RFJ3 edit. The learned correction changes with fault posture.')

sub('C. Implications for adaptation design')
p('For robot-cell design, select an acceptable nominal cost, compare fault recovery at the resulting operating points, and cover the lock postures expected in operation. The reference-fault results make a compact constant correction a useful starting baseline. The posture and two-finger tests identify where a richer correction retains higher success.')


h('VI. CONCLUSION')
p('Residual procedures retain a substantial reference little-finger recovery advantage after full-weight learning-rate refinement. This gain accompanies additional nominal loss and changes sharply with lock posture. A constant correction reproduces strong reference-lock recovery with a compact representation. Together, these results connect policy-update choices to the retention requirements and fault configurations of deployed robotic systems.')
h('ACKNOWLEDGMENT')
p('Claude (Anthropic) and ChatGPT/Codex (OpenAI) assisted with drafting and editing Sections I–VI and with analysis and plotting code. The authors reviewed the text, code and reported results.','figure caption')
q=p('¹ Data and code: https://github.com/woxihuanni99/Adapting-Pretrained-Dexterous-Grasping-Policies-to-Deployment-Shifts','figure caption')
q.paragraph_format.first_line_indent=Pt(0)
h('REFERENCES')
for i,key in enumerate(citations,1):
    if i==23:
        # Balance the final reference page without shrinking IEEE typography.
        b=D.add_paragraph();b.paragraph_format.space_before=Pt(0);b.paragraph_format.space_after=Pt(0)
        b.paragraph_format.line_spacing=Pt(1);b.paragraph_format.keep_with_next=True
        r=b.add_run();r.font.size=Pt(1);r.add_break(WD_BREAK.COLUMN)
    p(f'[{i}] '+BIB[key],'references')

D.core_properties.title=title
D.core_properties.author='Haoze Lan; Yongbin Yuan'
D.core_properties.subject='CSCWD 2027 conference submission manuscript'
D.core_properties.keywords='dexterous grasping; policy adaptation; recovery; retention'
D.core_properties.comments='Conference condensation; source: paper/v23/numbers.json and reports/cscwd_extension_1101/analysis/same_runtime/numbers.json. Original files preserved.'
D.save(DOCX)
# Remove inactive sample copyright text from the retained template package.
with ZipFile(DOCX) as z:parts={name:z.read(name) for name in z.namelist()}
for name in parts:
    if re.fullmatch(r'word/(footer|header)\d+\.xml',name):
        root=etree.fromstring(parts[name])
        for el in root.xpath('//*[local-name()="t"]'):el.text=''
        parts[name]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(DOCX,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
(OUT/'draft_text.txt').write_text('\n\n'.join(content),encoding='utf-8')
(OUT/'table_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'citation_map.json').write_text(json.dumps({i+1:{'source_paragraph':k,'reference':BIB[k]} for i,k in enumerate(citations)},ensure_ascii=False,indent=2),encoding='utf-8')
print('SAVED',DOCX,'words',len(re.findall(r'\S+',' '.join(content))),'references',len(citations))
