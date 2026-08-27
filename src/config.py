import numpy as np

COLUMN_NAMES = [
    "target", "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate",
    "personal_status_sex", "other_debtors", "present_residence_since", "property",
    "age", "other_installment_plans", "housing", "num_existing_credits", "job",
    "num_people_liable", "telephone", "foreign_worker",
]

N_QUBITS = 6
ENCODING_RANGE = (0, np.pi / 4)
N_PER_CLASS = 100
RANDOM_STATE = 42
DATA_PATH = "data/german_credit_data.csv"
ASSETS_DIR = "assets"
