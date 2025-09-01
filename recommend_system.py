import pandas as pd
import numpy as np
import re
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('assets/profile_symptom.csv')

def bin_age(a): return f"{(a//10)*10}s"
# ---------- helpers ----------
def is_nan(x):
    try: return math.isnan(x)  # np.nan, float('nan')
    except: return False

def clean_text(s: str) -> str:
    s = s.strip().lower()
    # unify spaces and dashes/underscores
    s = re.sub(r'[_\-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s

def nospace(s: str) -> str:
    return re.sub(r'\s+', '', s)

# Build bilingual synonym dictionary → canonical labels
SYNONYMS = {
    # respiratory
    "cough": ["cough", "ไอ", "ไอกลางคืน", "ไอมีเสมหะ", "ไอกลางคืนมีเสมหะ"],
    "sore_throat": ["sore throat", "เจ็บคอ", "คอแห้ง", "คอแดง", "กลืนเจ็บ", "เจ็บคอไข้", "ไอเจ็บคอ"],
    "runny_nose": ["runny nose", "runnynose", "น้ำมูกไหล", "จมูกตัน", "คัดจมูก", "แน่นจมูก",
                   "stuffynose", "nasal congestion", "nasalcongestion", "ไอคัดจมูก"],
    "phlegm": ["phlegm", "เสมหะ", "เสมหะไหลลงคอ", "มีเสมหะ", "มีเสมหะคัดจมูก", "มีเสมหะไอ",
               "มีเสมหะน้ำมูกไหล", "คันคอเสมหะไหลลงคอ"],
    "wheeze": ["wheezing", "หายใจมีเสียงวี๊ด"],
    "shortness_of_breath": ["labored breathing", "หายใจหอบเหนื่อย", "เหนื่อย", "หายใจไม่ออก",
                            "หายใจไม่สะดวก", "breathless onlying", "breathlessonlying"],
    "hoarseness": ["hoarseness", "เสียงแหบ"],
    # ENT – ear
    "ear_pain": ["ear pain", "earpain", "ปวดหู"],
    "ear_discharge": ["ear discharge", "eardischarge", "eardischargeearpain"],
    "hearing_loss": ["hearing loss", "hearingloss", "หูดับ", "การได้ยินลดลง"],
    "tinnitus": ["เสียงดังในหู"],
    # eyes
    "eye_pain": ["eye pain", "eyepain", "ปวดตา", "ปวดลูกตา", "ปวดตาเคืองตา"],
    "red_eye": ["ตาแดง"],
    "eye_discharge": ["eye discharge", "eyedischarge", "ขี้ตาเยอะ"],
    "itchy_eyes": ["itchy eyes", "คันตา", "เคืองตา", "eyeirritation"],
    "dry_eyes_skin": ["dry skin", "dryskin", "ตาแห้ง", "ผิวแห้ง"],
    "tearing": ["น้ำตาไหล"],
    "blurred_vision": ["blurry vision", "blurryvision", "ตาพร่ามัว", "ตามองเห็นไม่ชัด",
                       "มองเห็นภาพซ้อน"],
    "yellow_eyes_skin": ["ตาเหลือง", "yellowishskin"],
    # nose/bleeding
    "nosebleed": ["เลือดกำเดาไหล", "แน่นจมูกเลือดกำเดาไหล"],
    "sneeze": ["จาม", "จามบ่อย", "sneezing", "itchy nose sneezing", "itchynosesneezing"],
    # mouth
    "oral_ulcer": ["แผลในช่องปาก", "ฝ้าขาวที่ลิ้น", "มีแผล"],
    "lip_lesion": ["แผลริมฝีปาก"],
    # systemic / neuro
    "fever": ["ไข้", "fever", "ตัวร้อน", "ไข้คอแดง", "ร้อนวูบวาบไข้"],
    "headache": ["ปวดหัว", "headache", "ปวดขมับ", "headachecough"],
    "dizziness_lightheaded": ["เวียนศีรษะ", "มึนศีรษะ", "หน้ามืด", "ตาลายหน้ามืด",
                              "dizzy", "lightheaded", "dizzyblackout", "swaying", "บ้านหมุน"],
    "numbness_tremor_weakness": ["ชา", "มือสั่น", "มืออ่อนแรง", "แขนอ่อนแรง"],
    "fatigue": ["exertionfatique", "ง่วงตลอดเวลา"],
    # derm
    "itch": ["คัน", "itch", "genitalsitching"],
    "rash": ["ผื่น", "skinrash", "มีแผลผื่น"],
    "bruise": ["ฟกช้ำ", "จ้ำเลือด", "จุดเลือดออก"],
    "skin_lump": ["ก้อนที่ผิวหนัง", "skinlumpacne", "armpitlump", "ก้อนที่รักแร้",
                  "ก้อนที่หลังหู", "ก้อนที่ขา", "ก้อนที่ศีรษะ", "ก้อนบริเวณใบหน้า",
                  "ก้อนบริเวณขาหนีบ"],
    # GI
    "abdominal_pain": ["ปวดท้อง", "ปวดบั้นเอว", "ปวดสีข้าง", "ปวดท้องน้อย",
                       "stomachache", "backpain (if abdominal?)", "แสบท้อง", "เรอเปรี้ยว",
                       "เรอเปรี้ยวปวดท้อง", "จุกแน่นท้อง", "จุกแน่นท้องปวดท้อง"],
    "heartburn": ["แสบท้อง", "เรอเปรี้ยว", "drythroat (GERD?)"],
    "bloating": ["ท้องอืด"],
    "constipation": ["ท้องผูก", "decreased stool caliber", "decreasedstoolcaliber",
                     "อุจจาระลำเล็กลง"],
    "diarrhea": ["ท้องเสีย", "ถ่ายเหลว", "diarrhea", "ถ่ายปนเลือด", "ถ่ายเป็นเลือดสด"],
    "nausea_vomit": ["คลื่นไส้", "อาเจียน", "vomit", "คลื่นไส้อาเจียน", "อาเจียนคลื่นไส้"],
    "blood_in_stool": ["ถ่ายปนเลือด", "ถ่ายเป็นเลือดสด"],
    "loss_of_appetite": ["loss of appetite", "lossofappetite"],
    "weight_loss": ["weight loss stomachache", "weightlossstomachache"],
    # chest/resp overlap
    "chest_pain": ["เจ็บหน้าอก"],
    # urinary
    "dysuria": ["ปัสสาวะแสบขัด", "เจ็บเวลาปัสสาวะ", "ปัสสาวะเป็นเลือดปัสสาวะแสบขัด"],
    "hematuria": ["ปัสสาวะเป็นเลือด", "bloodyurine"],
    "frequency_urgency": ["ปัสสาวะบ่อย", "ปัสสาวะไม่สุด", "ปัสสาวะกะปริบกะปรอย",
                          "ปัสสาวะเล็ดราด", "strainingtourinate", "ปัสสาวะเป็นตะกอน", "ปัสสาวะขุ่น"],
    # musculoskeletal
    "back_pain": ["ปวดหลัง", "backpain"],
    "neck_pain": ["ปวดคอ", "ปวดต้นคอ", "neckpain", "ปวดคอปวดหัวไหล่", "ปวดต้นคอปวดบ่า"],
    "shoulder_pain": ["ปวดบ่า", "ปวดหัวไหล่", "ปวดบ่าปวดหัวไหล่"],
    "knee_pain": ["ปวดเข่า"],
    "ankle_pain": ["ปวดข้อเท้า", "anklepain"],
    "foot_pain": ["ปวดเท้า", "footpain"],
    "hand_wrist_pain": ["ปวดมือ", "ปวดข้อมือ", "ปวดนิ้วมือ", "ปวดข้อนิ้วมือ"],
    "elbow_pain": ["ปวดข้อศอก"],
    "rib_pain": ["ปวดซี่โครง"],
    "jaw_pain": ["ปวดกราม"],
    "joint_pain_swelling": ["ปวดข้อ", "jointpain", "บวม", "แขนบวม", "ขาบวม"],
    "muscle_ache": ["ปวดเมื่อยกล้ามเนื้อทั่วๆ", "ปวดขา", "ปวดแขน", "ปวดน่อง"],
    # throat swallowing speaking
    "odynophagia": ["pain on swallowing", "painonswallowing", "จุกแสบลำคอ"],
    "dysphagia": ["difficultyswallowing", "difficultyswallowingSorethroat", "กลืนลำบาก", "กลืนติด"],
    "dysphonia": ["difficultyspeaking"],
    # bleeding (nose/urine handled), bruises handled above
    # mental health (simple grouping)
    "insomnia": ["นอนไม่หลับ"],
    "anxiety_stress": ["วิตกกังวล", "เครียด", "รู้สึกไร้ค่า"],
    "self_harm_thoughts": ["คิดฆ่าตัวตาย", "ทำร้ายตัวเอง", "เก็บตัวอยากตาย"],
    # injuries
    "injury_trauma": ["กระแทก", "บาดเจ็บ", "animalbite", "รถล้ม"],
}

# Pre-compile regex patterns.
# For each synonym we match both the normal string and "no-space" version (to catch e.g. nasalcongestion).
def make_pattern(phrase):
    p = re.escape(phrase)
    p_ns = re.escape(nospace(phrase))
    # Thai doesn't respect \b well, so use a loose search; English use word-like boundaries.
    return re.compile(rf'({p}|{p_ns})', flags=re.IGNORECASE)

PATTERNS = {canon: [make_pattern(s) for s in syns] for canon, syns in SYNONYMS.items()}

def normalize_one(text):
    text = clean_text(text)
    if not text:
        return set()
    text_ns = nospace(text)
    labels = set()
    for canon, regs in PATTERNS.items():
        for rgx in regs:
            if rgx.search(text) or rgx.search(text_ns):
                labels.add(canon)
                break
    return labels

def normalize_symptom_list(items):
    out = set()
    for x in items:
        if x is None or is_nan(x): 
            continue
        out |= normalize_one(str(x))
    return sorted(out)

split_column = df['search_term'].str.split(',', expand=True)
split_column[split_column.columns] = split_column[split_column.columns].applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
split_column = split_column.applymap(lambda x: np.nan if x == '' else x)
split_column[split_column.columns] = split_column[split_column.columns].applymap(lambda lst: normalize_symptom_list(lst if isinstance(lst, (list, tuple)) else [lst]))
split_column = split_column.applymap(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "")

df['symptoms'] = split_column.values.tolist()
df['symptoms'] = df['symptoms'].apply(lambda x: [i for i in x if pd.notna(i) and str(i).strip() != ""])
df['ctx_items'] = df.apply(lambda r: [f"GENDER={r['gender'].lower()}", f"AGE={bin_age(r['age'])}"], axis=1)
df['basket'] = df['ctx_items'] + df['symptoms']

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