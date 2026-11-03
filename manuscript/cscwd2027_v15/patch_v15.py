"""Restore the audited RFJ3 correlation in the v14 abstract; no other edits."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import json,hashlib,shutil
R=Path(__file__).resolve().parent; B=R.parent; W=R/'output'; W.mkdir(exist_ok=True)
src=R.parent/'cscwd2027_v14/output/DexGrasp_CSCWD2027_1103_v14.docx'
assert src.exists(), 'Run the v14 patch first as described in README.md'
def package_sha(path):
    with ZipFile(path) as z:
        h=hashlib.sha256()
        for name in sorted(z.namelist()):h.update(name.encode()+b'\0'+hashlib.sha256(z.read(name)).digest())
        return h.hexdigest()
assert package_sha(src)=='36cc78c0c142eb483babaee79e107c604fd7a9edb04f60a5f0076ae958c536e1'
(W/'artifact.md').write_text('Narrow edit to retained v14: replace only the correlation clause in Abstract. Preserve all other paragraphs, tables, equations, figures, runs, styles, page geometry and package parts; preserve Times New Roman sizes and six-page IEEE format. Input SHA256 11ee4f16e2b1a13289b85858b09c12b2baa3139028f4e71134fac19cd7bc34d2. Render and view every final page. No numerical recomputation, simulation or training.',encoding='utf8')
with ZipFile(src) as z:parts={n:z.read(n) for n in z.namelist()}
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
root=etree.fromstring(parts['word/document.xml'])
paras=root.xpath('./w:body/w:p',namespaces=ns)
text=lambda p:''.join(p.xpath('.//w:t/text()',namespaces=ns))
p=next(p for p in paras if text(p).startswith('Abstract—'));old=text(p)
before='Joint-level diagnostics connect this recovery to a concentrated ring-finger target shift;'
after='Across 51 analyzed runs spanning all seven update variants, recovery rank-correlates with the ring-finger RFJ3 target change at ρ = 0.97;'
assert old.count(before)==1
new=old.replace(before,after);nodes=p.xpath('.//w:t',namespaces=ns);nodes[0].text=new
for n in nodes[1:]:n.text=''
parts['word/document.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
out=W/'DexGrasp_CSCWD2027_1103_v15.docx'
with ZipFile(out,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
change=dict(before=before,after=after,old_abstract_words=len(old.split()),new_abstract_words=len(new.split()),diagnostic_source='../../../extensions/cscwd2027_1102/review_v12/numbers.json',source_rho=.9654298642533936,new_experiments=0)
(W/'changes.json').write_text(json.dumps(change,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(change,ensure_ascii=False,indent=2));print(out)
