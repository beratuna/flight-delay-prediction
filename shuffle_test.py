import pandas as pd
import numpy as np
import csv
from sklearn.utils import shuffle

def main():
    arr = np.arange(0, 100)
    df = pd.DataFrame(arr, columns=["berat"])
    df.to_csv("./deneme.csv", index=False, encoding="utf-8")

    df = pd.read_csv("./deneme.csv")

    df = shuffle(df)
    df.to_csv("./shuffle_deneme.csv", index=False, encoding="utf-8")

main()