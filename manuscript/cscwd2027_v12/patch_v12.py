"""Targeted OOXML prose edits to the supplied v11; preserve figures and tables."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import json, hashlib, shutil

R=Path(__file__).resolve().parent;B=R.parent;W=R/'output';W.mkdir(exist_ok=True)
src=R/'reference_v11.docx'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='ec9ff39a53f62b9e848f13135442706888390c20666e16e70c2fcfea64f8730e'
if not (R/'reference_v11.docx').exists():shutil.copy2(src,R/'reference_v11.docx')
out=W/'DexGrasp_CSCWD2027_1103_v12.docx'
with ZipFile(src) as z:parts={n:z.read(n) for n in z.namelist()}
root=etree.fromstring(parts['word/document.xml']);ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
paras=root.xpath('./w:body/w:p',namespaces=ns)
def txt(p):return ''.join(p.xpath('.//w:t/text()',namespaces=ns))
changes=[]
def setp(start,new):
 matches=[p for p in paras if txt(p).startswith(start)];assert len(matches)==1,(start,len(matches))
 p=matches[0];nodes=p.xpath('.//w:t',namespaces=ns);old=txt(p);assert nodes
 nodes[0].text=new
 for x in nodes[1:]:x.text=''
 changes.append(dict(old=old,new=new))

setp('Abstract—','Abstract—Fault adaptation must recover grasping while preserving normal performance. We compare full-weight fine-tuning and frozen-planner residual updates with PPO and GRPO across eight simulated conditions on Shadow and Allegro hands, using about 19,000 grasp attempts per run. At the reference little-finger lock, residual updates reach 64–65% success and retain a 28.0-point advantage after learning-rate refinement; full-weight means are slightly higher in the other seven conditions. A constant 24-dimensional output offset, trained with 48 parameters, reaches 67.1% success at 4.1 points of nominal loss. Across 51 available diagnostic runs, recovery correlates strongly with the ring-finger RFJ3 target change (Spearman ρ = 0.97). Raw and σ₀-scaled constant offsets represent the same corrections but select different recovery–retention operating points. At the retrained 75% lock, RFJ3 edits are smaller and procedure means cluster at 41–43%. An update record makes recovery, retention, evaluation reference and correction size explicit for developer–integrator review.')
setp('The comparison gives a sharp answer.','The comparison gives a practical result. At the reference little-finger lock, the initially selected residual updates double full-weight recovery. A constant output offset also reaches 67.1% success, and RFJ3 target change is strongly associated with recovery across the available diagnostic checkpoints. The raw and scaled offset controls reach similar recovery at different nominal costs. At the 75% lock, the learned edits and relative performance change.')
setp('Our contributions are:','Our contributions are: (1) a run-level recovery–retention comparison of full-weight and residual updates, each with PPO and GRPO, across eight shift conditions on two hands; (2) compact-correction controls and joint-level diagnostics that characterize reference-lock recovery and its posture boundary; and (3) an update record that carries recovery, retention, evaluation reference and correction size from developer to integrator.')
setp('Training runs are the statistical units.','Training runs are the statistical units. Contrasts use two-sided 95% Welch intervals, with PPO and GRPO weighted equally in residual-minus-full-weight effects; operating-point Student t intervals condition on the fixed frozen reference. Table IV uses one-sided p-values with Holm correction. Curve comparisons linearly interpolate test means; the 10,000-draw percentile bootstrap (seed 110205) resamples paired fault/nominal runs within each rate and recomputes their shared loss range.')
setp('Reference LF gives the strongest separation.','Reference LF gives the strongest separation. The frozen planner achieves 16.9% success; full-weight procedures reach about 32% and residual procedures 64–65% (Table II). Full-weight means are slightly higher in the other seven conditions. Six of eight adapted Allegro means exceed five points of nominal loss. PPO–GRPO contrasts range from −0.3 to +1.8 pp. The GRPO pair shows a large LF gap without a value network: 32.1% for full-weight and 64.2% for residual updates.')
setp('Fig. 1 places recovery on a common nominal-cost axis.','Fig. 1 places recovery on a common nominal-cost axis. Averaged over the shared 2.3–5.4 pp loss range, the interpolated residual curves lie 17.7 pp above full-weight curves (95% bootstrap interval [14.9, 20.0]). At nearby observed costs, full-weight PPO reaches 50.2% success at 5.1 pp loss and residual PPO reaches 65.2% at 5.4 pp.')
setp('Fig. 1. Reference-LF','Fig. 1. Reference-LF recovery–retention curves over the initial grids. Small markers: training runs; lines: candidate means; stars: development choices. Shading marks the shared loss range; full-weight GRPO and residual PPO have no candidate mean inside it. The inset shows the full range; residual 10⁻² candidates are omitted.')
setp('Residual updates still lead by 28.0','Residual updates still lead by 28.0 [23.1, 33.0] pp at reference LF, with 3.1 pp more nominal loss; the PPO pair differs by 24.1 [14.2, 33.9] pp. Under PCL, full-weight updates lead by 2.7 [0.4, 5.1] pp. A fixed-scale, two-epoch residual GRPO control reaches 68.0% LF success at 3×10⁻⁴, but its 7.8 pp development loss exceeds the selection limit.')
setp('The residual advantage needs almost no capacity.','A constant correction provides strong reference-LF recovery. Replacing the residual network with a learnable offset c gives the deployed mean m₀(s) + σ₀ ⊙ c. Training c and 24 log-scales (48 parameters) reaches 67.1% success at 4.1 [3.8, 4.4] pp nominal loss (Table V); the source residual GRPO network reaches 64.2% at 4.9 pp loss. A 24-parameter offset with fixed sampling scale reaches 67.6%, and averaging an already-trained residual head over nominal states gives 66.7% without further training. The planner still adapts each grasp to the object; only the correction is constant (Fig. 3).')
setp('Fig. 3. Final frames','Fig. 3. Three deliberately selected Isaac Gym examples (seen objects, first runs). S/F: recorded outcome. Scaled shift: 24 parameters. Shown outcomes match the archive; two of nine rerenders differ on unshown objects.')
setp('The constant is enough only at the reference lock.','The tested simplification loses recovery under other fault settings. At retrained 75%, replacing the network by its mean changes success from 41.1% to 36.7%; under two-finger lock, the network, its mean and a directly trained constant reach 36.5%, 32.5% and 26.8%. The network retains more recovery in these controls.')
setp('B. One joint edit, priced by output coordinates','B. Joint-level shifts and output coordinates')
setp('Fig. 4 shows what the correction does.','Across 51 available reference-lock diagnostic runs, pooled over analyzed families and rates, success rank-correlates with RFJ3 target change at ρ = 0.97, versus 0.41 for the whole σ₀-normalized mean shift (Fig. 4). RFJ3 is the third ring-finger joint; its targets include tanh and joint limits.')
setp('Fig. 4. Lock success against','Fig. 4. RFJ3 target change on 2,048 common nominal states versus success. Colored: 51 available diagnostic runs, pooled for ρ; blue, full-weight/raw offsets; orange, frozen-planner corrections. Filled/hollow: selected/other rates. Gray: 12 runs retrained at 75%.')
setp('Update coordinates set the nominal price of this edit.','Constant-offset coordinates yield different observed nominal costs. For a frozen linear mean head m₀(s) = Wh₀(s) + b, a scaled offset d = σ₀ ⊙ c is exactly an output-bias change:')
setp('Raw and σ₀-scaled offsets therefore span','Raw and σ₀-scaled offsets span the same corrections. The selected raw offset moves RFJ3 by 0.15 rad and reaches 27.1% success, whereas the selected scaled offset moves it by 1.42 rad and reaches 67.6%. A larger raw rate reaches 1.47 rad and 68.7%, at 8.5 pp loss versus 4.5 pp for the scaled selection. With Adam and negligible numerical stabilizer, the scaled parameterization approximately multiplies effective output-coordinate steps by σ₀. RFJ3 is among the 12 largest-scale dimensions (σ₀ = 0.56). Residual GRPO allocates 87% of absolute mean change to those dimensions, versus 62% for full-weight GRPO, and moves RFJ3 by 1.29 rad versus 0.28 rad. Higher full-weight rates produce larger edits and costs, reaching 26.0 pp nominal loss for GRPO at 10⁻⁴. The compact controls demonstrate strong recovery with few adapted parameters and different selected costs across output coordinates.')
setp('At the 75% lock this route closes.','At the 75% lock the correction profile changes. Retrained models move RFJ3 by −0.07 to 0.68 rad and reach 37–44% success (Fig. 4, gray); residual GRPO has a 0.46 cosine with its reference-lock correction. The four procedure means lie at 41–43%.')
setp('The developer who trains an update and the integrator who deploys it','Model developers and cell integrators need a common description of the update. Following model cards [28], Table VI records trainable size, recovery, nominal loss and the evaluation batch defining the frozen reference. Residual GRPO has 4.9 pp loss against 91.0% frozen nominal success and 5.2 pp against 91.3% in the repeat batch; both loss intervals include five points. This makes the threshold sensitivity visible. The 48-parameter correction has its full loss interval below five points and deploys as 24 output offsets. Its RFJ3 change accounts for about two-thirds of summed absolute joint-target change on the common states. These fields connect model size, output effects and retention evidence in one record.')
setp('All results are simulated,','All results are simulated, with one open-loop planner family at the stated budgets. Locked joints hold targets; limp or torque-limited joints may differ. Three-run groups and sparse grids limit curve estimates. Test objects were explored before protocol lock, and LF/PCL evaluations reuse selection runs. A full-weight update with σ₀-scaled output coordinates would isolate the coordinate explanation; practitioner use of the update record remains untested.')
setp('At the reference little-finger lock, residual corrections','At the reference little-finger lock, residual corrections recover 64–65% of grasps against 32–41% for full-weight fine-tuning, and a large contrast remains after grid refinement. A 48-parameter constant correction reaches 67.1% success at 4.1 pp nominal loss. RFJ3 target change tracks reference-lock recovery across the available checkpoints; retraining at 75% produces smaller RFJ3 edits and procedure means of 41–43%. Compact corrections provide a useful reference-fault baseline whose recovery, retention and output changes can be reported directly.')

# Keep the equation's introduction with the editable equation across columns.
lead=next(p for p in paras if txt(p).startswith('Constant-offset coordinates yield'))
pp=lead.find('w:pPr',ns)
assert pp is not None
kn=pp.find('w:keepNext',ns)
if kn is None:
    kn=etree.Element('{'+ns['w']+'}keepNext')
    # pStyle precedes keepNext in CT_PPr; the retained paragraph has pStyle first.
    pp.insert(1 if len(pp) and etree.QName(pp[0]).localname=='pStyle' else 0,kn)
kn.set('{'+ns['w']+'}val','1')

# Balance the final reference columns after reexport with Times New Roman.
breaks=[p for p in paras if p.xpath('.//w:br[@w:type="column"]',namespaces=ns)]
assert len(breaks)==1
bp=breaks[0];target=next(p for p in paras if txt(p).startswith('[23] '))
parent=bp.getparent();parent.remove(bp);parent.insert(parent.index(target),bp)

parts['word/document.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(out,'w',ZIP_DEFLATED) as z:
 for n,data in parts.items():z.writestr(n,data)
(W/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf8')
(W/'main_text.txt').write_text('\n\n'.join(txt(p) for p in paras),encoding='utf8')
(W/'artifact.md').write_text('''# V11 fidelity contract for v12
Reference: review_1103_v12/reference_v11.docx, SHA256 ec9ff39a53f62b9e848f13135442706888390c20666e16e70c2fcfea64f8730e.
Supplied PDF inspected on all six pages in review_1103_v11/supplied-page-1.png through -6.png.
Preserve the supplied title, authors, IEEE Letter geometry, sections, 10pt Times New Roman body, 24pt title, 11pt author, 9pt abstract, 8pt captions/tables/references, figures and all table data. Preserve v11 corrected table-property order. Only selected body paragraph text, keepNext on the output-bias introduction, and relocation of the retained final reference column break before reference 23 in word/document.xml may change; all other ZIP parts remain byte equal. No new experiments, data filters, test families or causal mechanism claim. Reexport with the existing E-drive portable LibreOffice and inspect six final pages.
''',encoding='utf8')
print('Saved',out,'changed paragraphs',len(changes))
