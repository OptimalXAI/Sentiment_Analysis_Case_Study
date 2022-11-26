import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path,encoding='latin-1')
    df['Label'] = df['Ratings'].apply(lambda x: 1 if x >= 7 else (0 if x<=4 else 2))
    df=df[df.Label<2]
    data=df[['Reviews','Label']]
    return data