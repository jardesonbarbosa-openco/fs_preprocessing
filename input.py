import os
import pandas as pd

from feature_store.data_extraction.file_io import FileIO

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

fpath = os.path.join(DIR_PATH, 'data/input.parquet')

file_reader = FileIO()
df = file_reader.load(fpath, format='parquet', columns=['str_loan_uuid', 'str_user_cpf_x', 'scrcrdpnm6mmlv3_x', 'scrcrdcartaoaj_x', 'n_contratos_anteriores', 'max_atraso_anytime', 'if_fpd5_anytime'])

print(list(df))