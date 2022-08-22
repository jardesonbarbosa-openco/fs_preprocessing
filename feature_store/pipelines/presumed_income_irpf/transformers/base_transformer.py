from feature_store.data_preparation import text_handler, array_handler

import re
import pandas as pd
import numpy as np

def explode_dict_col(df: pd.DataFrame, dict_col='riskInfo'):
    '''Explodes risk_dict where each row is a
    tax report year.'''

    df = df.copy()

    df.loc[:, 'tax_report_data'] = (
        df[dict_col].apply(
            lambda x: x.values()))

    df = df.apply(pd.Series.explode).reset_index(drop=True).copy()

    return df

def get_irpf_status(df: pd.DataFrame, text_col: str):
    '''Receives pandas dataframe and column name and applies
    regex to column to generate new columns representing status
    of irpf application. Returns dataframe with new columns.'''

    df = df.copy()

    regex_not_consulted = re.compile(
        r'(?:^\s*$|\bdata\sde\snascimento\sinformada\b'
        r'.*\bestá\sdive|\bnão\scoletado'
        r'|\bocorreu\suma\sinconsistência\s?[.])'
        , re.IGNORECASE)

    regex_not_declared = re.compile(
        r'(?:\bconsta\sapresentação\sde\sdeclaração\sanual'
        r'\sde\sisento\b|\bapresentação\sda\sdeclaração\s'
        r'como\sisento\b|\bdeclaração\sconsta\scomo\sisento\b'
        r'|\bdeclaração\sconsta\scomo\spedido\sde'
        r'\sregularização\b|\bsua\sdeclaração\snão\sconsta'
        r'\sna\sbase\sde\sdados\b|\bainda\snão\sestá\sna'
        r'\sbase\b)', re.IGNORECASE)

    regex_tax_refund = re.compile(
        r'(?:\bsituação\sda\srestituição[:]\screditada\b'
        r'|\bsomente\sserá\spermitida\spor\smeio\sdo\scódigo\sde\sacesso\b'
        r'|\baguardando\sreagendamento\spelo\scontribuinte[.]?'
        r'|\bdevolvida\sà\sreceita\sfederal[,]?\sem\srazão\sdo\snão\sresgate\b'
        r'|\benviada\spara\scrédito\sno\sbanco\b'
        r'|\breagendada\spara\scrédito\sno\sbanco\b'
        r'|\bdados\sda\sliberação\sde\ssua\srestituição\b'
        r'|\bdeclaração\sestá\sna\sbase\sde\sdados\b'
        r'|\bestá\sna\sbase[,]\sutilize\so\sextrato\b'
        r'|\bdeclaração\sjá\sfoi\sprocessada[.]?$'
        r'|\brestituição[:]\saguardando\sdevolução\spelo\sbanco\b)'
        , re.IGNORECASE)
    
    col_list = ['irpf_extraction_error', 'irpf_not_declared', 'irpf_tax_refund']

    df.loc[:, 'irpf_extraction_error'] = text_handler.apply_regex_series(
        df[text_col], regex_not_consulted, handle_nan=False)
    df.loc[:, 'irpf_not_declared'] = text_handler.apply_regex_series(
        df[text_col], regex_not_declared)
    df.loc[:, 'irpf_tax_refund'] = text_handler.apply_regex_series(
        df[text_col], regex_tax_refund)

    df.loc[:, 'irpf_tax_to_pay'] = df[col_list].apply(
        lambda x: 1 not in x.values, axis=1).astype(int)

    return df

def retrieve_stars(y: int, x: int, star_arr: np.array):
    try:
        if y >= 16:
            stars = 5
        else:
            stars = star_arr[y][x]
    except IndexError:
        return -1
    else:
        return stars

def set_star_number(arr_declarations: np.array, arr_refunds: np.array):
    base_arr = [
        [0],
        [1, 1],
        [1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 2, 2],
        [1, 1, 2, 2, 3, 3],
        [1, 2, 2, 3, 3, 4, 4],
        [2, 2, 3, 3, 4, 4, 4, 5],
        [2, 3, 3, 4, 4, 4, 5, 5, 5],
        [2, 3, 4, 4, 4, 5, 5, 5, 5, 5],
        [3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5],
        [3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]

    max_len = np.array([len(arr) for arr in base_arr]).max()

    default_value = -1

    star_array = [
        np.pad(arr, (0, max_len - len(arr)),
        mode='constant',
        constant_values=default_value) for arr in base_arr]

    return np.array([retrieve_stars(y, x, star_array)
                        for y, x in zip(arr_declarations, arr_refunds)])


def get_presumed_income(year: int,
                        irpf_dict: dict,
                        branch_pl: str,
                        star_dict: dict,
                        year_list: list):

    year_d = str(array_handler.find_le(year_list, year))
    
    presumed_income_set = set()

    for key, value in irpf_dict.items():
        if int(value) > 7:
            value = '7'
        
        presumed_income_set.add(
            star_dict.get(year_d)
            .get(key)
            .get(value)
        )

    if int(irpf_dict.get('ESTR')) > 0:
        declared_branch_incm = (star_dict.get(year_d, {})
                            .get(branch_pl, {})
                            .get('1', 0))

        presumed_income_set.add(
            declared_branch_incm
        )

    return max(presumed_income_set)

def calculate_presumed_income(df: pd.DataFrame, income_dict: dict):

    year_list = [int(x) for x in list(income_dict.keys())]
    year_list.sort()

    branch_codes = ['year', 'ESTR', 'PERS', 'STIL',
                'PRIM', 'OUTR', 'HSBC', 'VANG',
                'UNIC', 'ESPA', 'PRIV', 'branch_code_pl']

    return df.loc[:, branch_codes].apply(
            lambda x: get_presumed_income(
                x[0],
                {'ESTR': str(x[1]),
                'PERS': str(x[2]),
                'STIL': str(x[3]),
                'PRIM': str(x[4]),
                'OUTR': str(x[5]),
                'HSBC': str(x[6]),
                'VANG': str(x[7]),
                'UNIC': str(x[8]),
                'ESPA': str(x[9]),
                'PRIV': str(x[10])},
                x[11],
                income_dict,
                year_list
            ), axis=1, raw=False)