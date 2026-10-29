import _paths
import numpy as np, pandas as pd, shapely, xml.etree.ElementTree as ET
from shapely.geometry import LineString
rng=np.random.default_rng(1)
lanes=[]
for _,el in ET.iterparse(_paths.A("ingolstadt.net.xml")):
    if el.tag=="edge" and el.get("type") in ("highway.motorway",):
        for ln in el.findall("lane"):
            g=LineString([tuple(map(float,p.split(","))) for p in ln.get("shape").split()])
            if g.length>600: lanes.append(g)
        el.clear()
print("motorway lanes >600m:",len(lanes))
rows=[]
for sid in range(120):
    L=lanes[rng.integers(len(lanes))]; v=rng.uniform(22,33)
    s0=rng.uniform(0,L.length-600); n=int(500/v)
    split=["train","validation","test"][sid%3]
    b=rng.uniform(0,40)*np.array([np.cos(a:=rng.uniform(0,2*np.pi)),np.sin(a)])
    for i in range(n):
        s=s0+v*i; p=L.interpolate(s); q=L.interpolate(s+1)
        hm=np.arctan2(q.y-p.y,q.x-p.x); compass=np.pi/2-hm
        noise=rng.normal(0,1.0,2); x,y=p.x+noise[0],p.y+noise[1]
        base=dict(split=split,receiver_id=999,sender_id=sid,message_id=sid*1000+i,timestamp=float(i),
                  speed=v,heading_sin=np.sin(compass),heading_cos=np.cos(compass))
        rows.append(dict(base,dataset_variant="benign",label=0,pos_x=x,pos_y=y,distance_to_road_edge=0.0))
        rows.append(dict(base,dataset_variant="constantPositionOffset",label=1,pos_x=x+b[0],pos_y=y+b[1],distance_to_road_edge=np.nan))
        bt=rng.uniform(-20,20,2)
        rows.append(dict(base,dataset_variant="randomPositionOffset",label=1,pos_x=x+bt[0],pos_y=y+bt[1],distance_to_road_edge=np.nan))
        rows.append(dict(base,dataset_variant="constantSpeedOffset",label=1,pos_x=x,pos_y=y,speed=v+5,distance_to_road_edge=0.0))
df=pd.DataFrame(rows)
# 감사 필드 흉내: 공격 위치의 도로 거리(아래 score 후 채움 없이 임의값)
df.loc[df.dataset_variant.str.contains("Position"),"distance_to_road_edge"]=rng.uniform(0,30,(df.dataset_variant.str.contains("Position")).sum())
df.to_parquet("synth.parquet",index=False); print(df.shape, df.groupby(["split","dataset_variant"]).size().unstack())
