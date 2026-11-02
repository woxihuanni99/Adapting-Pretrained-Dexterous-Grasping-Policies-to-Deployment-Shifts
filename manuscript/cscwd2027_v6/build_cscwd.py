from pathlib import Path
exec((Path(__file__).resolve().parent/'builder_prefix.py').read_text(encoding='utf8'),globals())
exec((R/('supplement_content.py' if SUPPLEMENT else 'main_content.py')).read_text(encoding='utf8'),globals())
D.core_properties.title=title
D.core_properties.author='Haoze Lan; Yongbin Yuan'
D.core_properties.subject='CSCWD 2027 supplementary material' if SUPPLEMENT else 'CSCWD 2027 manuscript'
D.core_properties.keywords='dexterous grasping; recovery; retention; policy adaptation'
D.save(DOCX)
with ZipFile(DOCX) as z:parts={name:z.read(name) for name in z.namelist()}
for name in parts:
    if re.fullmatch(r'word/(footer|header)\d+\.xml',name):
        root=etree.fromstring(parts[name])
        for el in root.xpath('//*[local-name()="t"]'):el.text=''
        parts[name]=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(DOCX,'w',ZIP_DEFLATED) as z:
    for name,data in parts.items():z.writestr(name,data)
kind='supplement' if SUPPLEMENT else 'main'
(OUT/(kind+'_text.txt')).write_text('\n\n'.join(content),encoding='utf8')
(OUT/(kind+'_table_audit.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
if not SUPPLEMENT:(OUT/'citation_map.json').write_text(json.dumps({i+1:{'source_paragraph':k,'reference':BIB[k]} for i,k in enumerate(citations)},ensure_ascii=False,indent=2),encoding='utf8')
print('SAVED',DOCX,'words',len(re.findall(r'\S+',' '.join(content))),'references',len(citations))
