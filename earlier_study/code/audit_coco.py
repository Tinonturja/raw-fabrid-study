#!/usr/bin/env python3
"""Check RAW-FABRID COCO annotation consistency, geometry and crop overlap."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def audit(d, is_patch):
    images=d['images']; anns=d['annotations']; ids=Counter(i['id'] for i in images); annids=Counter(a['id'] for a in anns)
    byid={i['id']:i for i in images}; byimage=Counter(a['image_id'] for a in anns); issues=Counter(); examples=defaultdict(list)
    def note(key, value):
        issues[key]+=1
        if len(examples[key])<5: examples[key].append(value)
    for a in anns:
        im=byid.get(a['image_id'])
        if im is None: note('annotation_missing_image',a['id']); continue
        w=256 if is_patch else im.get('width'); h=256 if is_patch else im.get('height')
        box=a.get('bbox',[])
        if len(box)!=4: note('bbox_invalid',a['id']); continue
        x,y,bw,bh=box
        if bw<=0 or bh<=0 or x<0 or y<0 or x+bw>w or y+bh>h: note('bbox_outside_image_or_nonpositive',a['id'])
        if a.get('area',0)<=0: note('nonpositive_area',a['id'])
        for poly in a.get('segmentation',[]):
            if len(poly)<6 or len(poly)%2: note('polygon_invalid_length',a['id']); continue
            if any(px<0 or px>=w for px in poly[0::2]) or any(py<0 or py>=h for py in poly[1::2]): note('polygon_out_of_bounds',a['id'])
    return {'image_rows':len(images),'unique_image_ids':len(ids),'duplicate_image_ids':sum(c-1 for c in ids.values() if c>1),
            'annotation_rows':len(anns),'unique_annotation_ids':len(annids),'duplicate_annotation_ids':sum(c-1 for c in annids.values() if c>1),
            'images_with_annotations':len(byimage),'issues':dict(issues),'examples':dict(examples)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('directory',type=Path); ap.add_argument('output',type=Path); a=ap.parse_args(); p=a.directory
    high=json.loads((p/'RAW_FABRID_HighRes_COCO.json').read_text()); patch=json.loads((p/'RAW_FABRID_Patches_COCO.json').read_text())
    hc=read_csv(p/'RAW_FABRID_HighRes_Metadata.csv'); pc=read_csv(p/'RAW_FABRID_Patches_Metadata.csv')
    hr={x['filename']:x for x in hc}; pr={x['patch_name']:x for x in pc}; issues=Counter(); ex=defaultdict(list); byparent=defaultdict(list)
    def note(k,v):
        issues[k]+=1
        if len(ex[k])<5: ex[k].append(v)
    for im in patch['images']:
        name=im['file_name']; c=pr.get(name)
        if c is None: note('patch_coco_missing_csv',name)
        elif c['parent_raw']!=im['parent'] or c['type']!=im['type']: note('patch_metadata_mismatch',name)
        if im['parent'] not in hr: note('patch_parent_missing',name)
        r=im['r_offset']; col=im['c_offset']
        if r<0 or r+256>1024 or col<0 or col+256>1792: note('crop_out_of_bounds',name)
        byparent[im['parent']].append(im)
    names={im['file_name'] for im in patch['images']}; missing=sorted(set(pr)-names)
    for name in missing: note('patch_csv_missing_coco',name)
    annotated={ann['image_id'] for ann in patch['annotations']}
    for im in patch['images']:
        if im['type']=='test_defect' and im['id'] not in annotated: note('defect_patch_without_coco_annotation',im['file_name'])
        if im['type']!='test_defect' and im['id'] in annotated: note('good_patch_with_coco_annotation',im['file_name'])
    same_source_overlap=Counter(); overlapped_test_good=set(); overlapped_test_defect=set()
    for parent, ims in byparent.items():
        good=[i for i in ims if i['type']=='test_good']; bad=[i for i in ims if i['type']=='test_defect']
        for g in good:
            for b in bad:
                h=max(0,min(g['r_offset']+256,b['r_offset']+256)-max(g['r_offset'],b['r_offset']))
                w=max(0,min(g['c_offset']+256,b['c_offset']+256)-max(g['c_offset'],b['c_offset']))
                if h*w:
                    same_source_overlap['good_defect_pairs']+=1; overlapped_test_good.add(g['file_name']); overlapped_test_defect.add(b['file_name'])
    hi_names={im['file_name'] for im in high['images']}; high_ids={im['id']:im['file_name'] for im in high['images']}; high_annotated={high_ids.get(z['image_id']) for z in high['annotations']}
    marked={r['filename'] for r in hc if r['has_defect'].lower()=='true'}
    result={'highres_coco':audit(high,False),'patch_coco':audit(patch,True),
            'highres_csv_missing_coco_count':len(set(hr)-hi_names),'highres_coco_missing_csv_count':len(hi_names-set(hr)),
            'highres_defect_marked_without_coco':sorted(marked-high_annotated),'highres_coco_annotated_not_defect_marked':sorted(high_annotated-marked),
            'metadata_consistency_issues':dict(issues),'examples':dict(ex),
            'within_test_crop_overlap':{'good_defect_pairs':same_source_overlap['good_defect_pairs'],
              'unique_test_good_patches_overlapping_test_defect_crop':len(overlapped_test_good),
              'unique_test_defect_patches_overlapping_test_good_crop':len(overlapped_test_defect)}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
