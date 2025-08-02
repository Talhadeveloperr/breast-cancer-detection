# etl/transform.py
def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values (just in case)
    df = df.dropna()

    # Rename target
    df['target'] = df['target'].map({0: 'malignant', 1: 'benign'})

    return df
