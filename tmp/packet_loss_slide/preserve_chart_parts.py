"""Preserve original native chart packages/workbooks after text-only slide edit.

Artifact Tool currently imports the values but drops externalData on export.
Copy the original chart parts rather than substituting workbook snapshots.
"""
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E

source, candidate, output = map(Path, sys.argv[1:])
with ZipFile(source) as original, ZipFile(candidate) as edited:
    chart_parts = {name for name in original.namelist() if '/charts/' in name or name.endswith('.xlsx')}
    if sum(name.endswith('.xlsx') for name in chart_parts) != 2:
        raise ValueError('Expected the two original embedded workbooks')
    def signatures(z):
        result=[]
        for name in sorted(z.namelist()):
            if '/charts/' in name and name.endswith('.xml') and '/_rels/' not in name:
                root=E.fromstring(z.read(name))
                result.append([[node.text for node in series.xpath('.//*[local-name()="xVal" or local-name()="yVal"]//*[local-name()="pt"]/*[local-name()="v"]')]
                               for series in root.xpath('//*[local-name()="ser"]')])
        return result
    if signatures(original) != signatures(edited):
        raise ValueError('Chart data changed during a text-only edit')
    types=E.fromstring(edited.read('[Content_Types].xml'))
    known={(E.QName(node).localname,node.get('Extension') or node.get('PartName')) for node in types}
    for node in E.fromstring(original.read('[Content_Types].xml')):
        identity=E.QName(node).localname,node.get('Extension') or node.get('PartName')
        if identity not in known:
            types.append(node)
            known.add(identity)
    with ZipFile(output,'w',ZIP_DEFLATED) as final:
        for info in edited.infolist():
            if info.filename in chart_parts or '/charts/' in info.filename or info.filename.endswith('.xlsx'):
                continue
            final.writestr(info,E.tostring(types,xml_declaration=True,encoding='UTF-8') if info.filename=='[Content_Types].xml' else edited.read(info.filename))
        for name in sorted(chart_parts):
            final.writestr(name,original.read(name))
print('Preserved both original native charts and embedded workbooks without changing values.')
