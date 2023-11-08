import os
import pandas as pd
from openpyxl import load_workbook

folder_path = './title/prep_xls'
prepared_file_folder = './title/prep_xls/result'
trash_file_folder = './title/prep_xls/result'

def check_row(row):
    if pd.notna(row['url']) and not ('.ru' in row['url'] or '.рф' in row['url']):
        return False
    if pd.notna(row['url']) and ('.ru' in row['url'] or '.рф' in row['url']):
        if len(row['url']) > 30:
            return False
    if pd.notna(row['title']):
        if isinstance(row['title'], str):
            if len(row['title']) < 20 or "Не удалось получить страницу" in row['title'] or "Не является html файлом" in row['title']:
                return False
    if pd.notna(row['description']):
        if isinstance(row['description'], str):
            if len(row['description']) < 20:
                return False
    return True

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        prepared_df = pd.DataFrame(columns=['url', 'title', 'description'])
        trash_df = pd.DataFrame(columns=['url', 'title', 'description'])

        file_path = os.path.join(folder_path, file_name)
        data = pd.read_excel(file_path)

        for index, row in data.iterrows():
            if check_row(row):
                prepared_df = pd.concat([prepared_df, pd.DataFrame([row])])
            else:
                trash_df = pd.concat([trash_df, pd.DataFrame([row])])

        prepared_file_path = os.path.join(prepared_file_folder, f'{os.path.splitext(file_name)[0]}_prepared.xlsx')
        trash_file_path = os.path.join(trash_file_folder, f'{os.path.splitext(file_name)[0]}_trash.xlsx')
        prepared_df.to_excel(prepared_file_path, index=False)
        trash_df.to_excel(trash_file_path, index=False)
