import pandas as pd
import numpy as np
from collections import defaultdict
from math import log
import ast
from utils import *

df_out = pd.read_csv('assets/Addition_process_profile.csv')
df_out["symptoms"] = df_out["symptoms"].apply(ast.literal_eval)
melt = df_out.explode('symptoms').copy()
melt['age_bin'] = melt['age'].apply(bin_age)

symptoms = sorted({s for row in df_out['symptoms'] for s in row})
idx = {s: i for i, s in enumerate(symptoms)}
n = len(symptoms)

co = np.zeros((n, n), dtype=np.int32)
freq = np.zeros(n, dtype=np.int32)
for row in df_out['symptoms']:
    ids = list({idx[s] for s in row if s in idx})
    for i in ids: freq[i] += 1
    for i in ids:
        for j in ids:
            if i != j:
                co[i, j] += 1

# -- Jaccard
jac = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    for j in range(n):
        if i == j: 
            continue
        denom = freq[i] + freq[j] - co[i, j]
        if denom > 0:
            jac[i, j] = co[i, j] / denom

melt = melt.reset_index().drop_duplicates(['index', 'gender', 'age_bin', 'symptoms'])

prior = (melt.groupby(['gender', 'age_bin', 'symptoms']).size()
         .reset_index(name='n'))
prior['p'] = prior.groupby(['gender', 'age_bin'])['n'].transform(lambda x: x / x.sum())
prior = prior.drop(columns='n')

def prior_vector(gender, age):
    ab = bin_age(age)
    sub = prior[(prior['gender'] == gender) & (prior['age_bin'] == ab)]
    v = np.zeros(n, dtype=np.float32)
    if not sub.empty:
        for s, p in zip(sub['symptoms'], sub['p']):
            if s in idx: 
                v[idx[s]] = p
    else:
        glob = (melt.groupby('symptoms').size() / len(melt)).reset_index(name='p')
        for s, p in zip(glob['symptoms'], glob['p']):
            if s in idx: 
                v[idx[s]] = p
    return v

# สร้างสถิติ slot -> P(slot=value | symptom)
slot_counts = {
    ('duration_category'): defaultdict(int), 
    ('previous_treatment'): defaultdict(int),
}
symptom_counts = defaultdict(int)

for _, r in melt.iterrows():
    s = r['symptoms']
    if s not in idx:
        continue
    symptom_counts[s] += 1
    dc = str(r.get('duration_category', '') or '').strip() or 'NA'
    pt = str(r.get('previous_treatment', '') or '').strip() or 'NA'
    slot_counts['duration_category'][(s, dc)] += 1
    slot_counts['previous_treatment'][(s, pt)] += 1

def logP_slot_given_symptom(symptom, slot_name, slot_value, eps=1.0):
    """log P(slot=value | symptom) with Laplace smoothing"""
    s = symptom
    if symptom_counts[s] == 0:
        return 0.0
    num = slot_counts[slot_name][(s, slot_value)] + eps
    # กำหนดจำนวนค่าที่เป็นไปได้ (vocab size) ต่อ slot แบบง่าย ๆ
    # ดึงชุดค่าที่เคยพบจริงจากตารางนับ
    observed_vals = {v for (sym, v), c in slot_counts[slot_name].items() if sym == s}
    V = max(1, len(observed_vals))
    den = symptom_counts[s] + eps * V
    return log(num / den)

def slot_score_vector(duration_category, previous_treatment, weights=None):
    """
    คืนเวกเตอร์ขนาด |symptoms| ของคะแนนความเข้ากันกับ 2 ช่อง
    ใช้ log P(duration|s) + log P(previous_treatment|s) (ถ้ามี)
    weights: เช่น {'duration_category': 1.0, 'previous_treatment': 1.0}
    """
    if weights is None:
        weights = {'duration_category': 1.0, 'previous_treatment': 1.0}
    dc = (duration_category or 'NA')
    pt = (previous_treatment or 'NA')
    out = np.zeros(n, dtype=np.float32)
    for i, s in enumerate(symptoms):
        ll = 0.0
        ll += weights.get('duration_category', 1.0) * logP_slot_given_symptom(s, 'duration_category', dc)
        ll += weights.get('previous_treatment', 1.0) * logP_slot_given_symptom(s, 'previous_treatment', pt)
        out[i] = ll
    return out

def rec_itemknn_with_slots(observed_symptoms, gender, age,
                           duration_category=None, previous_treatment=None,
                           k=5, alpha=0.6, beta=0.25, gamma=0.15,
                           weights=None):
    normalized = normalize_symptom_list(observed_symptoms)
    obs_ids = [idx[s] for s in normalized if s in idx]

    pv = prior_vector(gender, age)
    slotv = slot_score_vector(duration_category, previous_treatment, weights)

    if not obs_ids:   # cold start
        score = beta * pv + gamma * slotv
    else:
        sim = jac[obs_ids].sum(axis=0)
        # กันไม่ให้แนะนำซ้ำ
        for oid in obs_ids:
            sim[oid] = -1e9
        score = alpha * sim + beta * pv + gamma * slotv

    order = np.argsort(-score)
    recs = [symptoms[i] for i in order if symptoms[i] not in normalized][:k]
    return recs

gender = input('Enter your gender: ')
age = int(input('Enter your age: '))
symptom = input('Enter your symptom: ')
print('Duration: <1 day / <1 week / 1-3 weeks / 1-3 months / 3-6 months / >6 months')
dur = input('Enter your duration: ')
print('Previous treatment: ไม่เคย / เคยรักษามาก่อน / เคยทานยาเองแล้วไม่ดีขึ้น')
previous = input('Enter previous treatment: ')

rec = rec_itemknn_with_slots(observed_symptoms=[symptom], gender=gender, age=age,
                             duration_category=dur, previous_treatment=previous,
                             k=5, alpha=0.7, beta=0.15, gamma=0.15)

print('Top 5 of similar symptopms:')
for i in range(len(rec)):
    print(f'{i+1}.{rec[i]}')