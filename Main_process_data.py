import pandas as pd
import numpy as np
from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


df = pd.read_csv('assets/profile_symptom.csv')
split_column = df['search_term'].str.split(',', expand=True)
split_column[split_column.columns] = split_column[split_column.columns].applymap(lambda x: x.replace(" ", "") if isinstance(x, str) else x)
split_column = split_column.applymap(lambda x: np.nan if x == '' else x)
split_column[split_column.columns] = split_column[split_column.columns].applymap(lambda lst: normalize_symptom_list(lst if isinstance(lst, (list, tuple)) else [lst]))
split_column = split_column.applymap(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "")

df['symptoms'] = split_column.values.tolist()
df['symptoms'] = df['symptoms'].apply(lambda x: [i for i in x if pd.notna(i) and str(i).strip() != ""])
df['ctx_items'] = df.apply(lambda r: [f"GENDER={r['gender'].lower()}", f"AGE={bin_age(r['age'])}"], axis=1)
df['basket'] = df['ctx_items'] + df['symptoms']

df.to_csv('assets/Main_process_profile.csv', index=False)