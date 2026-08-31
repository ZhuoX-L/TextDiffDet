import argparse, json, math
from collections import defaultdict
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def xywh_to_xyxy(b):
    x,y,w,h=map(float,b); return np.array([x,y,x+w,y+h],float)
def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12)
def match(gt_by_img, pred_by_img, thr, iou_thr=.5):
    tp=fp=0; vals=[]
    for image_id in sorted(set(gt_by_img)|set(pred_by_img)):
        gs=gt_by_img.get(image_id,[]); used=set()
        ps=sorted((p for p in pred_by_img.get(image_id,[]) if p['score']>=thr),key=lambda z:-z['score'])
        for p in ps:
            pb=xywh_to_xyxy(p['bbox']); best=-1; bj=None
            for j,g in enumerate(gs):
                if j in used: continue
                v=iou(pb,xywh_to_xyxy(g['bbox']))
                if v>best: best=v; bj=j
            if bj is not None and best>=iou_thr:
                used.add(bj); tp+=1; g=gs[bj]; gb=xywh_to_xyxy(g['bbox'])
                pc=np.array([(pb[0]+pb[2])/2,(pb[1]+pb[3])/2]); gc=np.array([(gb[0]+gb[2])/2,(gb[1]+gb[3])/2])
                gw,gh=g['bbox'][2],g['bbox'][3]
                vals.append({'iou':best,'nce':float(np.linalg.norm(pc-gc)/(math.hypot(gw,gh)+1e-12)),
                             'area_error':abs(p['bbox'][2]*p['bbox'][3]-gw*gh)/(gw*gh+1e-12),'gt_area':gw*gh})
            else: fp+=1
    total_gt=sum(map(len,gt_by_img.values())); fn=total_gt-tp
    return tp,fp,fn,vals
def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0; f=2*p*r/(p+r) if p+r else 0
    return p,r,f
def coco_metrics(gt_file,pred_file):
    c=COCO(gt_file); d=c.loadRes(pred_file); e=COCOeval(c,d,'bbox'); e.evaluate(); e.accumulate(); e.summarize()
    return {'mAP50:95':e.stats[0],'AP50':e.stats[1],'AP75':e.stats[2]}
def group_ap50(gt, preds, lo, hi):
    # COCOeval area ranges, using fixed train-derived thresholds.
    c=COCO(); c.dataset=json.loads(json.dumps(gt)); c.createIndex(); d=c.loadRes(preds)
    e=COCOeval(c,d,'bbox'); e.params.iouThrs=np.array([.5]); e.params.areaRng=[[lo,hi]]; e.params.areaRngLbl=['custom']
    e.params.maxDets=[1,10,1000]; e.evaluate(); e.accumulate()
    prec=e.eval['precision']; x=prec[prec>-1]; ap=float(x.mean()) if len(x) else float('nan')
    return ap
def main():
    a=argparse.ArgumentParser(); a.add_argument('--train-gt',required=True); a.add_argument('--eval-gt',required=True); a.add_argument('--pred',required=True); a.add_argument('--out',required=True); z=a.parse_args()
    train=json.load(open(z.train_gt)); gt=json.load(open(z.eval_gt)); preds=json.load(open(z.pred))
    areas=np.array([x.get('area',x['bbox'][2]*x['bbox'][3]) for x in train['annotations']],float); q1,q2=np.quantile(areas,[1/3,2/3])
    g=defaultdict(list); p=defaultdict(list)
    for x in gt['annotations']: g[x['image_id']].append(x)
    for x in preds: p[x['image_id']].append(x)
    # A reproducible 0.01 grid avoids tuning to individual test-set scores.
    scores=np.linspace(0,1,101)
    best=(-1,None,None)
    for t in scores:
        tp,fp,fn,_=match(g,p,float(t)); P,R,F=prf(tp,fp,fn)
        if F>best[0]: best=(F,float(t),(tp,fp,fn,P,R))
    F,t,(tp,fp,fn,P,R)=best; _,_,_,vals=match(g,p,t)
    arr=lambda k:np.array([v[k] for v in vals],float)
    iv=arr('iou'); nv=arr('nce'); av=arr('area_error')
    groups={'small':(0,q1),'medium':(q1,q2),'large':(q2,float('inf'))}; strat={}
    for name,(lo,hi) in groups.items():
        sub={i:[x for x in xs if lo<=x.get('area',x['bbox'][2]*x['bbox'][3])<(hi if math.isfinite(hi) else 1e100)] for i,xs in g.items()}
        stp,sfp,sfn,_=match(sub,p,t); strat[name]={'AP50':group_ap50(gt,preds,lo,hi if math.isfinite(hi) else 1e10),'Recall':stp/(stp+sfn) if stp+sfn else None,'TP':stp,'FN':sfn}
    out={'dataset':{'images':len(gt['images']),'gt_boxes':len(gt['annotations']),'pred_boxes':len(preds)},'coco':coco_metrics(z.eval_gt,z.pred),
      'threshold_selection_note':'F1-optimal threshold on this evaluation set; use only as validation threshold, not final test estimate.',
      'f1_optimal_threshold':t,'fixed_threshold_metrics':{'TP':tp,'FP':fp,'FN':fn,'Precision':P,'Recall':R,'F1':F,'FP_per_image':fp/len(gt['images'])},
      'localization_on_TP':{'n':len(vals),'Mean_IoU':float(iv.mean()),'Median_IoU':float(np.median(iv)),'IoU_IQR':[float(np.quantile(iv,.25)),float(np.quantile(iv,.75))],
       'IoU_ge_0.5_fraction':float((iv>=.5).mean()),'IoU_ge_0.75_fraction':float((iv>=.75).mean()),'Mean_NCE':float(nv.mean()),'Median_NCE':float(np.median(nv)),
       'Mean_AreaError':float(av.mean()),'Median_AreaError':float(np.median(av))},
      'train_area_tertiles':{'q33':float(q1),'q67':float(q2)},'size_stratified':strat,
      'patient_level':'not computable: no patient_id field or image-to-patient mapping in COCO annotations'}
    json.dump(out,open(z.out,'w'),ensure_ascii=False,indent=2); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
