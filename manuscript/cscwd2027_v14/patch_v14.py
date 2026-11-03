"""V13-based prose revision; preserve every table, figure and non-body part."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import hashlib, json, shutil

R=Path(__file__).resolve().parent; B=R.parent
W=R/'output'; W.mkdir(exist_ok=True)
src=R/'reference_v13.docx'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='982a40a4b327272814ae2d374921e3721b8d593acae220594b82b0974d6b9e05'
if not (R/'reference_v13.docx').exists(): shutil.copy2(src,R/'reference_v13.docx')
out=W/'DexGrasp_CSCWD2027_1103_v14.docx'
with ZipFile(src) as z: parts={n:z.read(n) for n in z.namelist()}
root=etree.fromstring(parts['word/document.xml'])
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paras=root.xpath('./w:body/w:p',namespaces=ns)
def txt(p): return ''.join(p.xpath('.//w:t/text()',namespaces=ns))
changes=[]
def setp(start,new):
    matches=[p for p in paras if txt(p).startswith(start)]
    assert len(matches)==1,(start,len(matches))
    p=matches[0]; nodes=p.xpath('.//w:t',namespaces=ns); assert nodes
    changes.append(dict(old=txt(p),new=new))
    nodes[0].text=new
    for n in nodes[1:]: n.text=''

# Recompute the newly reported development losses from the original ledger.
ledger_path=R/'original_selection.json'
j=json.loads(ledger_path.read_text(encoding='utf8'))
selection={}
for key in ['full_ppo','full_grpo','res_ppo','res_grpo']:
    m=j['methods'][key]; rate=m['d05']['shared_lr']
    selection[key]=dict(rate=rate,**{c:dict(loss_pp=-100*m['table'][rate][c]['dN'],n=len(m['table'][rate][c]['seeds'])) for c in ['LF','PCL']})
if not (R/'original_selection.json').exists():
    shutil.copy2(ledger_path,R/'original_selection.json')
loss_pairs='; '.join('/'.join(f"{selection[k][c]['loss_pp']:.1f}" for c in ['LF','PCL']) for k in selection)

setp('Abstract—', 'Abstract—Fault adaptation must recover grasping while preserving normal performance. We show that strong recovery from a reference little-finger lock does not require a state-dependent adaptation network. Across eight simulated conditions on Shadow and Allegro hands, we compare full-weight and residual updates with PPO and GRPO using about 19,000 grasp attempts per run. Residual updates reach 64–65% reference-lock success against about 32% for full-weight updates, retaining a 28.0-point advantage after learning-rate refinement. A constant 24-dimensional correction, trained with 48 parameters, reaches 67.1% success at 4.1 points of nominal loss. Joint-level diagnostics connect this recovery to a concentrated ring-finger target shift; raw and scaled coordinates attain similar recovery at different retention costs. Posture and two-finger tests expose the limits of the constant correction. Compact corrections thus provide a concrete capacity baseline, and an update record makes their recovery, retention and output changes inspectable by developers and integrators.')
setp('The comparison gives a clear result.', 'The comparison gives a clear result. At the reference little-finger lock, the initially selected residual updates double full-weight recovery. A constant 24-dimensional correction reaches 67.1% success, establishing that this recovery does not require a state-dependent adaptation head. Across 51 analyzed runs, recovery tracks the ring-finger RFJ3 target change. Raw and scaled offsets attain similar recovery at different retention costs, while posture and two-finger tests show where the constant simplification loses ground.')
setp('Our contributions are:', 'Our contributions are: (1) a run-level recovery–retention comparison of four update procedures across eight shift conditions on two hands; (2) evidence that a constant output correction suffices for strong reference-lock recovery, with joint-level diagnostics and tests of its fault-posture boundary; and (3) an update record and review procedure that connect recovery, retention, evaluation reference and correction size for developer–integrator decisions.')
old19=txt(paras[19])
setp('The six Shadow conditions are', old19.replace('The primary setting keeps the released collision filter, which disables robot–table contact in half the environments; a separate check enables all robot–table collisions and adds a ground plane.', 'We retain DemoGrasp’s contact-tolerant setting, which disables robot–table contact in half the environments to let fingers reach beneath flat objects [3]. Section IV-D checks all robot–table collisions with a ground plane.'))
old27=txt(paras[27])
setp('The development set has 64 objects', old27+' Selected LF/PCL development losses are '+loss_pairs+' pp for full-weight PPO, full-weight GRPO, residual PPO and residual GRPO, respectively.')
setp('Test evaluation uses the policy mean', 'Selected checkpoints are evaluated on test objects using the policy mean and seed 4242. A run’s score averages seen-category success (141 instances, 1,128 attempts) and unseen-category success (78 instances in the 100 slots of the released list, 800 attempts). Nominal loss is frozen nominal success minus adapted nominal success in the same evaluation batch.')
setp('Reference LF gives the strongest separation.', 'Reference LF gives the strongest separation. The frozen planner achieves 16.9% success; full-weight procedures reach about 32% and residual procedures 64–65% (Table II). Full-weight means are slightly higher in the other seven conditions. Allegro reuses Shadow-selected rates without an Allegro development search; six of eight adapted Allegro means exceed five points of nominal loss. PPO–GRPO contrasts range from −0.3 to +1.8 pp. The critic-free GRPO pair also shows a large LF gap: 32.1% for full-weight and 64.2% for residual updates.')
old37=txt(paras[37])
setp('Fig. 1. Reference-LF', old37.replace('no candidate mean inside it','no candidate mean in its strict interior'))
old48=txt(paras[48])
h4=' H4 applies each single-shift update unchanged after adding a second shift. Gains over the corresponding frozen baseline decrease by 11.7 pp for LF → LF+PCL, 3.4 pp for PCL → LF+PCL and 1.4 pp for TH → TH+ZR; the ZR → TH+ZR reduction is −0.3 [−1.8, 1.2] pp.'
setp('KL regularization does not close the gap.',old48+h4)
setp('Effects and unadjusted two-sided',txt(paras[50]).replace('Effects and unadjusted','Initial-batch effects and unadjusted'))
setp('V. WHAT THE CORRECTION CHANGES','V. COMPACT CORRECTION ANALYSIS')
setp('A constant correction matches the residual network', 'Strong reference-lock recovery needs no state-dependent adaptation head. Replacing the residual network with a learnable offset c gives the deployed mean m₀(s) + σ₀ ⊙ c. Training c and 24 log-scales (48 parameters) reaches 67.1% success at 4.1 [3.8, 4.4] pp nominal loss (Table V), compared with 64.2% at 4.9 pp loss for the source residual GRPO network. A 24-parameter offset with fixed sampling scale reaches 67.6%, and averaging an already-trained residual head over nominal states gives 66.7% without further training. The frozen planner still adapts each grasp to the object; only the correction is constant (Fig. 3).')
setp('Fig. 3. Final frames', 'Fig. 3. Isaac Gym simulation renders of three selected reference-LF episodes (seen-category objects, first training run per procedure). S/F: recorded evaluation outcome. Scaled shift: the 24-parameter σ₀-scaled offset.')
setp('Raw and σ₀-scaled offsets span', 'Raw and σ₀-scaled offsets span the same corrections. At the selected rates, 3×10⁻³ (raw) and 3×10⁻² (scaled), they move RFJ3 by 0.15 and 1.42 rad and reach 27.1% and 67.6% success. A larger raw rate reaches 1.47 rad and 68.7%, at 8.5 pp nominal loss versus 4.5 pp for the scaled selection. Adam normalizes the σ₀-scaled gradients, leaving output steps approximately proportional to ησ₀; the tenfold selected-rate increase outweighs RFJ3’s scale of 0.56. Residual GRPO allocates 87% of absolute mean change to the 12 largest-scale dimensions, versus 62% for full-weight GRPO, and moves RFJ3 by 1.29 rad versus 0.28 rad. Higher full-weight rates produce larger edits and costs, reaching 26.0 pp nominal loss for GRPO at 10⁻⁴. The compact controls separate capacity from operating point: both coordinates achieve high recovery with the same 24 parameters, but at different nominal costs.')
setp('Model developers and cell integrators need', 'The update record supports three review steps. First, the developer and integrator specify the target fault, tested postures and nominal-loss budget. Second, the developer reports run-level recovery and loss, their intervals, the frozen reference, trainable size and joint-target changes (Table VI), following model cards [28]. Third, the integrator compares the measured loss with the agreed budget and checks whether the tested postures cover the intended use. Under the study’s five-point mean-loss rule, residual GRPO lies at 4.9 pp initially and 5.2 pp on repeat; both intervals span five. The record makes this threshold sensitivity visible. The 48-parameter correction reaches 67.1% recovery with its loss interval entirely below five points and deploys as 24 offsets; RFJ3 contributes about two-thirds of its absolute joint-target change. These fields make the recovery–retention choice and the physical output edit explicit in the same review.')
setp('All results are simulated,', 'All results are simulated with one open-loop planner family at the stated budgets. Locked joints hold targets; limp or torque-limited joints may differ. Three-run groups and sparse grids limit curve estimates. Test objects were explored before protocol lock. A full-weight update with σ₀-scaled output coordinates would directly test the coordinate explanation for the network comparison. Practitioner use of the update record remains to be evaluated.')
setp('At the reference little-finger lock, residual corrections', 'Strong recovery from the reference little-finger lock does not require a state-dependent adaptation network. A constant 24-dimensional correction, trained with 48 parameters, reaches 67.1% success at 4.1 pp nominal loss. Joint-level diagnostics show where the output changes concentrate, while coordinate controls expose different retention costs for similar recovery. Posture and two-finger tests show where the constant correction loses ground. Compact corrections should serve as a capacity baseline: added adaptation complexity should deliver measurable gains on the target fault. The update record gives developers and integrators a concrete account of that choice.')
for num in [3,17]:
    p=next(p for p in paras if txt(p).startswith(f'[{num}] ')); old=txt(p)
    import re
    new=re.sub(r', arXiv:\d{4}\.\d{4,5}\.', '.', old)
    assert new!=old
    setp(f'[{num}] ',new)

# Rebalance the retained reference break after the changed prose pagination.
breaks=[p for p in paras if p.xpath('.//w:br[@w:type="column"]',namespaces=ns)]
assert len(breaks)==1
bp=breaks[0]; target=next(p for p in paras if txt(p).startswith('[18] '))
parent=bp.getparent(); parent.remove(bp); parent.insert(parent.index(target),bp)

parts['word/document.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for n,data in parts.items(): z.writestr(n,data)
(W/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf8')
(W/'main_text.txt').write_text('\n\n'.join(txt(p) for p in paras),encoding='utf8')
(W/'artifact.md').write_text('''# V13 fidelity contract for v14
Reference: ../review_1103_v14/reference_v13.docx, SHA256 982a40a4b327272814ae2d374921e3721b8d593acae220594b82b0974d6b9e05. Its six pages were rendered and viewed in review_1103_v14/reference_render.
Preserve the supplied IEEE Letter page geometry and columns, title/author block, sections, Times New Roman body10pt/title24pt/author11pt/abstract9pt/table-caption-reference8pt, all six numerical tables, four figure images and editable equations. Preserve corrected table property ordering and all ZIP parts except word/document.xml byte for byte. Only unique matched body paragraph text is editable; the existing reference column break and equation keepNext may be relocated if actual six-page render requires it. Shorten prose before any layout change; never shrink typography. No new experiment or test family. Verify all final pages, table/part equality, input SHA and PDF fonts.
''',encoding='utf8')
(W/'numbers.json').write_text(json.dumps(dict(original_selection_source_SHA256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),frozen_dev=j['N0_frozen_2048'],selected_development=selection,RFJ3_scale=.5558702945709229,raw_selected_rate=.003,scaled_selected_rate=.03,scaled_to_raw_Adam_output_multiplier=.03*.5558702945709229/.003,new_training=0,new_simulation=0),indent=2),encoding='utf8')
print('Saved',out,'changed paragraphs',len(changes),'development loss pairs',loss_pairs)
