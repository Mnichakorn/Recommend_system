import pandas as pd
import numpy as np
import json
import unicodedata
import re
import ast
import math
from utils import *
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df = pd.read_csv('assets/profile_symptom.csv')
df["summary"] = df["summary"].apply(json.loads)
df["yes_symptoms"] = df["summary"].apply(lambda x: x.get("yes_symptoms", []))

split_column = df['search_term'].str.split(',', expand=True)
split_column[split_column.columns] = split_column[split_column.columns].applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
split_column = split_column.applymap(lambda x: np.nan if x == '' else x)

df['symptoms'] = split_column.values.tolist()
df["symptoms"] = df["symptoms"].apply(lambda x: [s for s in x if s is not None])
df["symptoms"] = df["symptoms"].apply(lambda x: [s for s in x if s not in [None, np.nan] and pd.notna(s)])

def safe_parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)  # Try literal_eval first
        except:
            try:
                return json.loads(x)  # Try json.loads if JSON-like
            except:
                return x  # Return as-is if both fail
    return x  # Already parsed

# Apply safe parsing
df["yes_symptoms"] = df["yes_symptoms"].apply(safe_parse)

# 2) Extract texts and answers
df['text'] = df['yes_symptoms'].apply(lambda lst: [d.get('text') for d in lst if isinstance(d, dict)])
df['answers'] = df['yes_symptoms'].apply(lambda lst: [d.get('answers') for d in lst if isinstance(d, dict)])
df["answers"] = df["answers"].apply(safe_parse)
df['previous_treatment'] = df["answers"].apply(lambda x: x[-1] if x else None)

def clean_previous(v):
    if isinstance(v, list) and len(v) > 0:
        val = v[0]
        # remove leading "การรักษาก่อนหน้า" or "Previous treatment"
        val = re.sub(r'^(การรักษาก่อนหน้า|Previous treatment)\s*', '', val)
        return val.strip()
    return None

df['previous_treatment'] = df['previous_treatment'].apply(clean_previous)
df["answers"] = df["answers"].apply(lambda x: x[:-1] if x else x)
df["text"] = df["text"].apply(lambda x: x[:-1] if isinstance(x, list) else x)

# Ensure columns are lists
def as_list(x):
    return x if isinstance(x, list) else ([] if pd.isna(x) else [x])

df["text"] = df["text"].apply(as_list)
df["answers"] = df["answers"].apply(as_list)

# Pair text[i] with answers[i] row-wise
def make_pairs(row):
    t, a = row["text"], row["answers"]
    m = min(len(t), len(a))   # in case lengths differ
    return list(zip(t[:m], a[:m]))

df["pairs"] = df.apply(make_pairs, axis=1)
# Explode so each pair is a row, Split tuple back into columns, Drop helper column
df = df.explode("pairs", ignore_index=True)
df[["text", "answers"]] = pd.DataFrame(df["pairs"].tolist(), index=df.index)
df = df.drop(columns=["pairs"])

# ---------- 1) ตัวช่วยทำความสะอาด ----------
def _clean(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()

def _safe_list(x):
    # รองรับทั้ง NaN, สตริงเดี่ยว, และลิสต์
    if isinstance(x, list):
        return [a for a in x if isinstance(a, str) and a.strip()]
    if isinstance(x, str) and x.strip():
        return [x]
    return []

# ---------- 2) regex สำหรับพาร์สไทย/อังกฤษ ----------
re_duration_th = re.compile(r"(?:ระยะเวลา)\s*([^\n\r\|]+)")
re_duration_th_alt = re.compile(r"(น้อยกว่า\s*\d+\s*วัน|ไม่เกิน\s*\d+\s*(?:วัน|สัปดาห์)|\d+\s*-\s*\d+\s*(?:วัน|สัปดาห์))")
re_duration_en = re.compile(r"\bDuration\b\s*([^\n\r\|]+)", re.IGNORECASE)

re_character_th = re.compile(r"(?:ลักษณะ|ปวด|กำลังอยาก|บ้านเอียง|sensation|characteristic)\s*([^\n\r\|]+)")

re_severity_th = re.compile(r"(?:ระดับ|severity|Severity)\s*([^\n\r\|]+)")
re_grade_en    = re.compile(r"\bGrade\b\s*([^\n\r\|]+)", re.IGNORECASE)

re_atk_th = re.compile(r"(?:ประวัติ\s*ATK|ATK)\s*([^\n\r\|]+)")
re_atk_en = re.compile(r"\bHistory\s*ATK\b\s*([^\n\r\|]+)", re.IGNORECASE)

re_covid_th = re.compile(r"(?:ประวัติใกล้ชิดผู้ป่วยโควิด-?19)\s*([^\n\r\|]+)")
re_covid_en = re.compile(r"\bHistory close contact to COVID-?19\b\s*([^\n\r\|]+)", re.IGNORECASE)

re_location_th = re.compile(r"(?:บริเวณ|ในลำคอ|ตำแหน่ง|location|Location)\s*([^\n\r\|]+)")

def _map_atk(x: str) -> str:
    s = x.lower()
    if "ยังไม่ได้ทำ" in s or "ไม่ได้ทำ" in s or "haven't done" in s or "not done" in s:
        return "not_done"
    if "ลบ" in s or "neg" in s:
        return "negative"
    if "บวก" in s or "pos" in s:
        return "positive"
    if s.strip() in {"yes", "no"}:   # เผื่อผู้ใช้ตอบสั้นๆ
        return "not_done" if s == "no" else "done"
    return _clean(x)

def _map_covid(x: str) -> str:
    s = x.lower()
    if "ไม่ได้ใกล้ชิด" in s or s.strip() == "no":  return "no"
    if "ใกล้ชิด" in s or s.strip() == "yes":        return "yes"
    return _clean(x)

# ---------- 3) ฟังก์ชันแยก answers -> slots ----------
def parse_answers_to_slots(answers: List[str]) -> Dict[str, str]:
    slots = {
        "duration": "no data",
        "characteristic": "no data",
        "atk_status": "no data",
        "covid_exposure": "no data",
        "location": "no data",
        "severity": "no data",
        "grade": "no data",
        "details_raw": "no data",
    }
    L = _safe_list(answers)
    if not L:
        return slots

    joined = " | ".join(_clean(a) for a in L)
    slots["details_raw"] = joined if joined else "no data"

    # duration
    m = re_duration_th.search(joined) or re_duration_en.search(joined)
    if m:
        slots["duration"] = _clean(m.group(1))
    else:
        m2 = re_duration_th_alt.search(joined)
        if m2:
            slots["duration"] = _clean(m2.group(1))

    # characteristic
    m = re_character_th.search(joined)
    if m:
        slots["characteristic"] = _clean(m.group(1))

    # severity / grade
    m = re_severity_th.search(joined)
    if m:
        slots["severity"] = _clean(m.group(1))
    m2 = re_grade_en.search(joined)
    if m2:
        slots["grade"] = _clean(m2.group(1))

    # ATK
    m = re_atk_th.search(joined) or re_atk_en.search(joined)
    if m:
        slots["atk_status"] = _map_atk(m.group(1))

    # COVID exposure
    m = re_covid_th.search(joined) or re_covid_en.search(joined)
    if m:
        slots["covid_exposure"] = _map_covid(m.group(1))

    # location
    m = re_location_th.search(joined)
    if m:
        slots["location"] = _clean(m.group(1))

    # เติม "no data" ให้ครบกรณีว่าง
    for k, v in list(slots.items()):
        if not isinstance(v, str) or not v.strip():
            slots[k] = "no data"
    return slots

# ---------- 4) ใช้กับ DataFrame ที่มีคอลัมน์ 'answers' ----------
# สมมติ df มีคอลัมน์: 'text' และ 'answers'
# df_slots จะเป็น DataFrame ใหม่ของคอลัมน์ slot ทั้งหมด
df_slots = df["answers"].apply(parse_answers_to_slots).apply(pd.Series)

# รวมกลับเข้า df เดิม
df_out = pd.concat([df.drop(columns=[], errors="ignore"), df_slots], axis=1)

# (ออปชัน) ให้แน่ใจว่าไม่มี NaN ตกค้าง
df_out = df_out.fillna("no data")

df_out["duration_category"] = df_out["duration"].apply(categorize_duration)
df_out['text'] = df_out['text'].apply(lambda lst: normalize_symptom_list(lst if isinstance(lst, (list, tuple)) else [lst]))
df_out = df_out[(~df_out["text"].apply(lambda x: isinstance(x, list) and len(x) == 0))]
df_out = df_out[['gender', 'age', 'text', 'previous_treatment','duration_category']]
df_out.rename(columns={'text':'symptoms'}, inplace=True)
df_out['previous_treatment'] = np.where(df_out['previous_treatment'] == "No, i've never got any treatment for this condition before.", 'ไม่เคย', 
                                        np.where(df_out['previous_treatment'] == "Yes, I've got some treatment from another doctor before.", 'เคยรักษามาก่อน', 
                                                 np.where(df_out['previous_treatment'] == "Yes, I used to take medicine by myself and it didn't get better.", 'เคยทานยาเองแล้วไม่ดีขึ้น', df_out['previous_treatment'])))
df_out.to_csv('assets/Addition_process_profile.csv', index=False)