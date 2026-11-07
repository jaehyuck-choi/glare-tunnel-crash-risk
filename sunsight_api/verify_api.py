"""API가 10 노트북 결과와 같은지 대조. 실행 위치: sunsight_api 폴더.
python verify_api.py ../sunsight_video/data   (시험표_mapillary.csv · 등급2단계_사진별.csv · tunnel_photos/ 가 있는 폴더)"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score
from model3_core import ratio_from_L20, grade, LAB
from fastapi.testclient import TestClient
from api import app
from model3_core import FEATURES
D = sys.argv[1].rstrip('/') + '/'
c = TestClient(app); print(c.get('/health').json())
mp = pd.read_csv(D + '시험표_mapillary.csv', dtype={'image_id': str}); ref = pd.read_csv(D + '등급2단계_사진별.csv', dtype={'image_id': str})
col = [x for x in ref.columns if x.startswith('주 모델')][0]
te = mp[mp.test_ok & (mp.cls == '입구')]
# ① 숫자 입력 (/predict/features) 66장
body = [{'id': x['image_id'], 'features': {k: x[k] for k in FEATURES if not pd.isna(x[k])}, 'sun_alt': x['sun_alt'],
         'cloud': None if pd.isna(x['cloud']) else x['cloud']} for _, x in te.iterrows()]
res = pd.DataFrame(c.post('/predict/features', json=body).json()).merge(ref[['image_id', col]], left_on='id', right_on='image_id')
print('① 숫자 입력: 10 노트북과 등급 일치', int((res.grade == res[col]).sum()), '/', len(res))
g = pd.Series(grade(ratio_from_L20(10 ** te.y.values, te.cls.values)), index=te.image_id.values)   # 정답 = 천리안 일사량 → L20 → 휘도비 → 등급
gt = g.loc[res.id].values
print('   API 입구 Macro F1', round(f1_score(gt, res.grade, average='macro', labels=LAB), 3), '(10 노트북 .844)',
      '| 상 재현', round(float(((res.grade == '상') & (gt == '상')).sum() / (gt == '상').sum()), 2), '| 정답 분포', pd.Series(gt).value_counts().to_dict())
# ② 사진 입력 (/predict/photo): 사진에서 피처·해 높이를 API가 직접 계산
n = ok = 0; worst = 0
for _, r in te.iterrows():
    p = D + f"tunnel_photos/{r['image_id']}.jpg"
    try: f = open(p, 'rb')
    except FileNotFoundError: continue
    data = {'captured_at': str(r['captured_at']), 'lat': r['lat'], 'lon': r['lon']}
    if not pd.isna(r['cloud']): data['cloud'] = r['cloud']
    o = c.post('/predict/photo', files={'image': f}, data=data).json()
    worst = max([worst] + [abs(o['photo_features'][k] - r[k]) for k in FEATURES if not pd.isna(r[k])])
    n += 1; ok += o['grade'] == ref.loc[ref.image_id == r['image_id'], col].iloc[0]
print('② 사진 입력: 등급 일치', ok, '/', n, '| 피처 최대 차이', round(worst, 5))
