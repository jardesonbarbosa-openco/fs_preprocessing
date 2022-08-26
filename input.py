import os
import pandas as pd

from feature_store.fs_engineering.presumed_income_irpf.transformer import PresumedIncomeIrpf

pipeline = PresumedIncomeIrpf()

pipeline.execute()

pipeline.export_output()