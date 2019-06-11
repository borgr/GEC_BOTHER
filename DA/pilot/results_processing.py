import pandas as pd
import numpy as np
from scipy import stats

RESULTS_FILE_ADDR = "Batch_3548123_batch_results.csv"


# gets an HIT id and return an array of size 100 with the indexes of the sentences in the hit
def get_indexes(df,HIT_id):
    return np.array(
        [df[col][HIT_id] for col in df if col.startswith('Input.Nucle_ID')])


# gets an HIT id and return a bolean array of size 100 with True in the indexes of the perfect sentences in the hit
def get_perfects(df,HIT_id):
    return np.array(
        [df[col][HIT_id] for col in df if col.startswith('Input.is_perfect')])


# gets an HIT id and return a bolean array of size 100 with True in the indexes of the control sentences in the hit
def get_control(df,HIT_id):
    return np.array(
        [df[col][HIT_id] for col in df if col.startswith('Input.is_control')])


def get_scores(df,HIT_id):
    return np.array([df[col][HIT_id] for col in df if col.startswith('Answer.mistakeRate')])

def get_z_scores(df,HIT_id):
    return stats.zscore(get_scores(df,HIT_id))


# return an array which contains this c_sentence ranks (array size depends on the number of annotators for the sentence)
def c_sentence_scores(df,id):
    pass  # TODO:  this.


# return an array of size 100 which contains this HIT scores
def HIT_scores(df,HIT_id):
    res = []
    for i in range(1, HIT_SIZE + 1):
        # add this HIT score for sentence i to res.
        pass


# return an array of size 15 which contains this HIT perfect sentences scores
def HIT_p_scores(df,HIT_id):
    pass  # TODO:  this.


def c_sentence_average(df,id):
    return average(c_sentence_scores(id))


def HIT_average(df,HIT_id):
    return average(HIT_scores(id))


def HIT_p_average(df,HIT_id):
    return average(HIT_p_scores(id))


# Python program to get average of a list
def average(lst):
    return sum(lst) / len(lst)


def main():
    pass



def parse_data(path):
    df = pd.read_csv(path, index_col=0)
    cols = sorted(df.columns.tolist())
    return df[cols]


if __name__ == "__main__":
    df = parse_data(RESULTS_FILE_ADDR)
    print(get_scores(df,0))


