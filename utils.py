import pandas as pd
import math
import re

def bin_age(a): 
    return f"{(a//10)*10}s"

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
                            "หายใจไม่สะดวก", "breathless onlying", "breathlessonlying", 
                            "shortnessofbreath", "shortnessofbreathwhenlyingdown"],
    "hoarseness": ["hoarseness", "เสียงแหบ"],
    # ENT – ear
    "ear_pain": ["ear pain", "earpain", "ปวดหู"],
    "ear_discharge": ["ear discharge", "eardischarge", "eardischargeearpain"],
    "hearing_loss": ["hearing loss", "hearingloss", "หูดับ", "การได้ยินลดลง", "หูอื้อ"],
    "tinnitus": ["เสียงดังในหู", "เสียงดังรบกวนในหู"],
    # eyes
    "eye_pain": ["eye pain", "eyepain", "ปวดตา", "ปวดลูกตา", "ปวดตาเคืองตา"],
    "eyelid_twitch": ["ตากระตุก"], 
    "red_eye": ["ตาแดง"],
    "eye_discharge": ["eye discharge", "eyedischarge", "ขี้ตาเยอะ", "ขี้ตา"],
    "itchy_eyes": ["itchy eyes", "คันตา", "เคืองตา", "eyeirritation"],
    "dry_eyes_skin": ["dry skin", "dryskin", "ตาแห้ง", "ผิวแห้ง"],
    "tearing": ["น้ำตาไหล"],
    "blurred_vision": ["blurry vision", "blurryvision", "ตาพร่ามัว", "ตามองเห็นไม่ชัด",
                       "มองเห็นภาพซ้อน"],
    "yellow_eyes_skin": ["ตาเหลือง", "yellowishskin", "jaundice"],
    # nose/bleeding
    "nose_pain": ["ปวดจมูก"], 
    "nosebleed": ["เลือดกำเดาไหล", "แน่นจมูกเลือดกำเดาไหล"],
    "sneeze": ["จาม", "จามบ่อย", "sneezing", "itchy nose sneezing", "itchynosesneezing"],
    # mouth
    "oral_ulcer": ["แผลในช่องปาก", "ฝ้าขาวที่ลิ้น", "มีแผล", "แผลในปาก"],
    "lip_lesion": ["แผลริมฝีปาก"],
    # systemic / neuro
    "fever": ["ไข้", "fever", "ตัวร้อน", "ไข้คอแดง", "ร้อนวูบวาบไข้"],
    "headache": ["ปวดหัว", "headache", "ปวดขมับ", "headachecough"],
    "dizziness_lightheaded": ["เวียนศีรษะ", "มึนศีรษะ", "หน้ามืด", "ตาลายหน้ามืด","lightheadness",
                              "dizzy", "lightheaded", "dizzyblackout", "swaying", "บ้านหมุน"],
    "numbness_tremor_weakness": ["ชา", "มือสั่น", "มืออ่อนแรง", "แขนอ่อนแรง","แขนขาอ่อนแรง"],
    "fatigue": ["exertionfatique", "ง่วงตลอดเวลา", "fatigueassociatedwithexertion", "ง่วงเยอะ"],
    # derm
    "acne": ["acne"],
    "itch": ["คัน", "itch", "genitalsitching"],
    "rash": ["ผื่น", "skinrash", "มีแผลผื่น"],
    "bruise": ["ฟกช้ำ", "จ้ำเลือด", "จุดเลือดออก"],
    "skin_lump": ["ก้อนที่ผิวหนัง", "skinlumpacne", "armpitlump", "ก้อนที่รักแร้",
                  "ก้อนที่หลังหู", "ก้อนที่ขา", "ก้อนที่ศีรษะ", "ก้อนบริเวณใบหน้า",
                  "ก้อนบริเวณขาหนีบ"],
    # GI
    "abdominal_pain": ["ปวดท้อง", "ปวดบั้นเอว", "ปวดสีข้าง", "ปวดท้องน้อย", "abdominalpain",
                       "stomachache", "backpain (if abdominal?)", "แสบท้อง", "เรอเปรี้ยว",
                       "เรอเปรี้ยวปวดท้อง", "จุกแน่นท้อง", "จุกแน่นท้องปวดท้อง"],
    "heartburn": ["แสบท้อง", "เรอเปรี้ยว", "drythroat (GERD?)", "drythroat"],
    "bloating": ["ท้องอืด"],
    "constipation": ["ท้องผูก", "decreased stool caliber", "decreasedstoolcaliber",
                     "อุจจาระลำเล็กลง", "narrowstool"],
    "diarrhea": ["ท้องเสีย", "ถ่ายเหลว", "diarrhea", "ถ่ายปนเลือด", "ถ่ายเป็นเลือดสด"],
    "nausea_vomit": ["คลื่นไส้", "อาเจียน", "vomit", "คลื่นไส้อาเจียน", "อาเจียนคลื่นไส้", "nausea"],
    "blood_in_stool": ["ถ่ายปนเลือด", "ถ่ายเป็นเลือดสด", "ถ่ายเป็นเลือด"],
    "loss_of_appetite": ["loss of appetite", "lossofappetite"],
    "weight_loss": ["weight loss stomachache", "weightlossstomachache","weightloss"],
    # chest/resp overlap
    "chest_pain": ["เจ็บหน้าอก"],
    # urinary
    "dysuria": ["ปัสสาวะแสบขัด", "เจ็บเวลาปัสสาวะ", "ปัสสาวะเป็นเลือดปัสสาวะแสบขัด"],
    "hematuria": ["ปัสสาวะเป็นเลือด", "bloodyurine", "bloodinurine"],
    "frequency_urgency": ["ปัสสาวะบ่อย", "ปัสสาวะไม่สุด", "ปัสสาวะกะปริบกะปรอย",
                          "ปัสสาวะเล็ดราด", "strainingtourinate", "ปัสสาวะเป็นตะกอน", "ปัสสาวะขุ่น"],
    # musculoskeletal
    "back_pain": ["ปวดหลัง", "backpain", "ปวดเอว"],
    "neck_pain": ["ปวดคอ", "ปวดต้นคอ", "neckpain", "ปวดคอปวดหัวไหล่", "ปวดต้นคอปวดบ่า"],
    "shoulder_pain": ["ปวดบ่า", "ปวดหัวไหล่", "ปวดบ่าปวดหัวไหล่", "ปวดไหล่"],
    "knee_pain": ["ปวดเข่า"],
    "ankle_pain": ["ปวดข้อเท้า", "anklepain"],
    "foot_pain": ["ปวดเท้า", "footpain"],
    "hand_wrist_pain": ["ปวดมือ", "ปวดข้อมือ", "ปวดนิ้วมือ", "ปวดข้อนิ้วมือ"],
    "elbow_pain": ["ปวดข้อศอก"],
    "rib_pain": ["ปวดซี่โครง"],
    "jaw_pain": ["ปวดกราม"],
    "joint_pain_swelling": ["ปวดข้อ", "jointpain", "บวม", "แขนบวม", "ขาบวม"],
    "muscle_ache": ["ปวดเมื่อยกล้ามเนื้อทั่วๆ", "ปวดขา", "ปวดแขน", "ปวดน่อง", "ปวดเมื่อยกล้ามเนื้อ"],
    # throat swallowing speaking
    "odynophagia": ["pain on swallowing", "painonswallowing", "จุกแสบลำคอ"],
    "dysphagia": ["difficultyswallowing", "difficultyswallowingSorethroat", "กลืนลำบาก", "กลืนติด"],
    "dysphonia": ["difficultyspeaking"],
    # bleeding (nose/urine handled), bruises handled above
    # mental health (simple grouping)
    "insomnia": ["นอนไม่หลับ"],
    "anxiety_stress": ["วิตกกังวล", "เครียด", "รู้สึกไร้ค่า"],
    "self_harm_thoughts": ["คิดฆ่าตัวตาย", "ทำร้ายตัวเอง", "เก็บตัวอยากตาย", "คิดอยากฆ่าตัวตายทำร้ายตนเอง"],
    # injuries
    "injury_trauma": ["กระแทก", "บาดเจ็บ", "animalbite", "รถล้ม","animalrelatedinjury", "historyoftrauma"],
    "bone_pain": ["ปวดกระดูก"],    
    # women’s health
    "vaginal_discharge": ["ตกขาวผิดปกติ"],
    "hot_flashes": ["ร้อนวูบวาบ"],
    # neuro/cardiac-like
    "syncope": ["วูบ", "fainted"],
    # hair
    "hair_loss": ["ผมร่วง"],
    "history_hypertension": ["ประวัติความดันสูง", "historyofhypertension(highbloodpressure)"],
    "history_gastritis": ["ประวัติโรคกระเพาะ"],
    "history_hyperlipidemia": ["ประวัติไขมันสูง"],
    "axillary_mass": ["axillarymass", "ก้อนบริเวณหู"],
    "floaters": ["จุดดำลอยในตา"],
    "imbalance": ["unsteady,lossofbalance", "เดินเซทรงตัวไม่ได้"],
    "dysarthria": ["slurredspeech"],
    "menorrhagia": ["ประจำเดือนมากกว่าปกติ"],
    "oligomenorrhea": ["ประจำเดือนมาน้อย,ประจำเดือนขาด"],
    "facial_pain": ["ปวดบริเวณใบหน้า"]
}

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

def categorize_duration(s: str):
    if not isinstance(s, str):
        return "unknown"
    s = s.lower().strip()

    # จัดการกรณีพิเศษ (ภาษาอังกฤษ)
    if "hour" in s or "24" in s or "น้อยกว่า 1 วัน" in s or "ไม่เกิน 1 วัน" in s or "less then a day" in s:
        return "<1 day"
    if re.search(r"less\s*than\s*a\s*day?", s):
        return "<1 day"
    if re.search(r"1[-–]3\s*วัน|1-3 days?|1[-–]3days?|4[-–]7\s*วัน|3-7\s*วัน|ตั้งแต่ 1 วัน ถึง 1 สัปดาห์|น้อยกว่า 10 วัน|4[-–]7\s*days?|less than 10days|less\s*than\s*3\s*days?|1[-–]7\s*day?|less then a week?", s):
        return "<1 week"
    if re.search(r"1-3\s*สัปดาห์|ไม่เกิน 1 สัปดาห์|less than a week|8[-–]14\s*วัน|1-2\s*สัปดาห์|มากกว่า 7 วัน|1[-–]3\s*weeks?", s):
        return "1-3 weeks"
    if re.search(r"1\s*เดือน|1-3\s*เดือน|10\s*[-–]\s*90\s*วัน|3-8\s*สัปดาห์|3-8\s*weeks?|มากกว่า\s*8\s*สัปดาห์.*2\s*เดือน", s):
        return "1-3 months"
    if re.search(r"3-6\s*เดือน", s):
        return "3-6 months"
    if re.search(r"มากกว่า 6\s*เดือน|more than 6months", s):
        return ">6 months"

    return "no specific"