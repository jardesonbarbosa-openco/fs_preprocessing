import os

from feature_store.data_extraction.file_io import FileIO

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

def get_irpf_df():
    fpath = os.path.join(DIR_PATH, '../data/input.csv')
    return file_reader.load(fpath,
                        sep=';',
                        parse_dates=['time_stamp'],
                        dtype={'bank_code_pl': str,
                                'branch_number_pl': str,
                                'loan_id': str})

def get_bank_df():
    fpath = os.path.join(DIR_PATH, '../data/bank.parquet')
    df = file_reader.load(fpath)
    df = df.rename(columns={
        'BankName': 'bank',
        'Codigo_Banco': 'bank_code'})
    return df

def get_branch_df():
    fpath = os.path.join(DIR_PATH, '../data/bank_branch.parquet')
    branch_df = file_reader.load(fpath)
    branch_df = branch_df.rename(columns={
        'Bank': 'bank_code',
        'Branch': 'branch'})

    return branch_df

def load_dataset(dataset):
    datasets = {
            'irpf': get_irpf_df,
            'bank_names': get_bank_df,
            'branch': get_branch_df
    }
    
    return datasets.get(dataset, None)()

    




file_reader = FileIO()