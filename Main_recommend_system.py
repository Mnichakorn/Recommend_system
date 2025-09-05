import pandas as pd
import numpy as np
import re
import math
import ast
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from utils import *

df = pd.read_csv('assets/Main_process_profile.csv')
df["symptoms"] = df["symptoms"].apply(ast.literal_eval)
melt = df.explode('symptoms')
melt['age_bin'] = melt['age'].apply(bin_age)

# symptom universe
symptoms = sorted({s for row in df['symptoms'] for s in row})
idx = {s:i for i,s in enumerate(symptoms)}
n = len(symptoms)

# co-occurrence & frequency
co = np.zeros((n,n), dtype=np.int32)
freq = np.zeros(n, dtype=np.int32)
for row in df['symptoms']:
    ids = list({idx[s] for s in row if s in idx})
    for i in ids: freq[i] += 1
    for i in ids:
        for j in ids:
            if i!=j: co[i,j] += 1

# Jaccard similarity
jac = np.zeros((n,n), dtype=np.float32)
for i in range(n):
    for j in range(n):
        if i==j: continue
        denom = freq[i] + freq[j] - co[i,j]
        if denom>0: jac[i,j] = co[i,j] / denom

melt = melt.reset_index().drop_duplicates(['index','gender','age_bin','symptoms'])
# demographic priors P(symptom | gender, age_bin)
prior = (melt.groupby(['gender','age_bin','symptoms']).size().reset_index(name='n'))
prior['p'] = prior.groupby(['gender','age_bin'])['n'].transform(lambda x: x / x.sum())
prior = prior.drop(columns='n')

def prior_vector(gender, age):
    ab = bin_age(age)
    sub = prior[(prior['gender']==gender) & (prior['age_bin']==ab)]
    v = np.zeros(n, dtype=np.float32)
    if not sub.empty:
        for s, p in zip(sub['symptoms'], sub['p']):
            if s in idx: v[idx[s]] = p
    else:
        # fallback to global marginal
        glob = (melt.groupby('symptoms').size() / len(melt)).reset_index(name='p')
        for s, p in zip(glob['symptoms'], glob['p']):
            if s in idx: v[idx[s]] = p
    return v

def rec_itemknn(observed_symptoms, gender, age, k=5, alpha=0.7):
    normalized_symptoms = normalize_symptom_list(observed_symptoms)
    obs_ids = [idx[s] for s in normalized_symptoms if s in idx]
    if not obs_ids:  # cold start: just return by prior
        pv = prior_vector(gender, age)
        order = pv.argsort()[::-1]
        return [symptoms[i] for i in order][:k]

    sim = jac[obs_ids].sum(axis=0)
    sim[obs_ids] = 0  # don’t recommend what’s already present
    pv = prior_vector(gender, age)
    score = alpha*sim + (1-alpha)*pv

    order = score.argsort()[::-1]
    recs = [symptoms[i] for i in order if symptoms[i] not in normalized_symptoms][:k]
    return recs