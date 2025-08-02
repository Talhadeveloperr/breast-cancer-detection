# etl/load.py
from .extract import extract_data
from .transform import clean_data


def load_data():
    df_raw = extract_data()
    df_clean = clean_data(df_raw)
    return df_clean

# Test it
if __name__ == '__main__':
    df = load_data()
    print(df.head())
