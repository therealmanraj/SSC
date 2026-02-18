import msoffcrypto
import pandas as pd
import io

import os
from dotenv import load_dotenv

load_dotenv()

password = os.getenv("PASSWORD")
encrypted_file = 'SSC - Full report - UPDATED.xlsx'
decrypted_file = io.BytesIO()

with open(encrypted_file, 'rb') as f:
    file = msoffcrypto.OfficeFile(f)
    file.load_key(password=password)
    file.decrypt(decrypted_file)

# Read all sheets into a dict of {sheet_name: DataFrame}
decrypted_file.seek(0)
sheets = pd.read_excel(decrypted_file, sheet_name=None)

for name, df in sheets.items():
    print(f"--- {name} ---")
    print(df.head())
    df.to_csv(f'data/{name}')
