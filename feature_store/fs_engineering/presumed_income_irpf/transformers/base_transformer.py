from feature_store.data_preparation import text_handler, array_handler

import re
import pandas as pd
import numpy as np

def explode_dict_col(df: pd.DataFrame,
                    dict_col='riskInfo',
                    tax_report_col_name='tax_report_data') -> pd.DataFrame:
    """
    Explodes risk_dict where each row is a
    tax report year.
    Args:
        df (pd.DataFrame): Dataframe containing dict column with tax data.
        dict_col (str): Name of column to explode.
        tax_report_col_name (str): Name of new column to receive exploded data.
    Returns:
        pd.DataFrame: New dataframe with exploded col as rows.
    """

    df = df.copy()

    df.loc[:, tax_report_col_name] = (
        df[dict_col].apply(
            lambda x: x.values()))

    df = df.apply(pd.Series.explode).reset_index(drop=True).copy()

    return df

def get_irpf_status(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Applies regex to column to generate new columns representing status
    of IRPF application.
    Args:
        df (pd.DataFrame): Dataframe with column to apply regex.
        text_col (str): Name of column to apply regex.
    Returns:
        pd.DataFrame: Dataframe with IRPF status as columns
    """

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

def retrieve_stars(num_declarations: int,
                    num_refunds: int,
                    star_arr: np.array) -> int:
    """
    Retrives number of stars of a single IRPF application based on number
    of declarations and tax refunds.
    Args:
        num_declarations (int): Number of times IRPF declared.
        num_refunds (int): Number of times tax refunded.
        star_arr: Matrix of stars where y is number of declarations and x
        is number of refunds.
    Returns:
        int: Number of stars of application
    """

    try:
        if num_declarations >= 16:
            stars = 5
        else:
            stars = star_arr[num_declarations][num_refunds]
    except IndexError:
        return -1
    else:
        return stars

def set_star_number(arr_declarations: np.array,
                    arr_refunds: np.array) -> np.array:
    """
    Retrieve number of stars based on array of declarations and array of refunds.
    Args:
        arr_declarations (int): Array of umber of times IRPF declared.
        arr_refunds (int): Array of umber of times tax refunded.
    Returns:
        np.array: Array of number of IRPF stars.
    """

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
                        year_list: list) -> np.array:
    """
    Get the presumed income of a single CPF for a loan application
    based on past IRPF data.
    Args:
        year (int): Year of loan application.
        irpf_dict (dict): Dict of past IRPF declarations
        based on bank branchs.
        branch_pl (str): Code of bank branch of loan application.
        star_dict (dict): Base dict of stars to retrieve presumed income
        based on brank branch.
        year_list (list): List of years present in star_dict.
    Returns:
        np.array: Array of number of IRPF stars.
    """
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

def calculate_presumed_income(df: pd.DataFrame,
                                income_dict: dict) -> pd.Series:
    """
    Calculate presumed income of a dict of cpfs with one-hot encoded
    IRPF declarations and refunds.
    Args:
        df (pd.DataFrame): Dataframe with data to calculate presumed income.
        income_dict (dict): Base dict of stars to retrieve presumed income
        based on brank branch.
    Returns:
        pd.Series: Pandas Series with presumed income per CPF.
    """

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