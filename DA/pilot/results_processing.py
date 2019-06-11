import pandas as pd
import numpy as np
from scipy import stats
RESULTS_FILE_ADDR = "Batch_3548123_batch_results.csv"


# gets an HIT id and return an array of size 100 with the indexes of the sentences in the hit
def get_indexes(df,HIT_id):
    return np.array(
        [df[col][HIT_id] for col in df if col.startswith('Input.Nucle_ID')])


# gets an HIT id and return a bolean array of size 100 with True in the indexes of the perfect sentences in the hit
def get_perfect_idxs(df, HIT_id):
    return np.array(
        [df[col][HIT_id] for col in df if col.startswith('Input.is_perfect')])



# gets an HIT id and return a bolean array of size 100 with True in the indexes of the control sentences in the hit
def get_control_idxs(df, HIT_id):
    return np.array(
        [df[col][HIT_id] for col in df if col.startswith('Input.is_control')])

# return an array of size 100 which contains this HIT scores
def get_scores(df,HIT_id):
    return np.array([df[col][HIT_id] for col in df if col.startswith('Answer.mistakeRate')])

# return an array of size 100 which contains this HIT z-scores
def get_z_scores(df,HIT_id):
    return stats.zscore(get_scores(df,HIT_id))

# return an array which contains this c_sentence ranks (array size depends on the number of annotators for the sentence)
def c_sentence_scores(df,id):

    pass  # TODO:  this.

# return an array of size 15 which contains this HIT perfect sentences scores
def HIT_p_scores(df,HIT_id):
    return get_z_scores(df,HIT_id)[get_perfect_idxs(df, HIT_id)]


def c_sentence_average(df,id):
    return np.mean(c_sentence_scores(id))

# returns the average z-score of perfect sentences
def HIT_p_average(df,HIT_id):
    return np.mean(HIT_p_scores(df,HIT_id))


def main():
    pass



def parse_data(path):
    df = pd.read_csv(path, index_col=0)
    cols = sorted(df.columns.tolist())
    return df[cols]


if __name__ == "__main__":


    df = parse_data(RESULTS_FILE_ADDR)
    controls = {}
    c_table = pd.DataFrame()
    c = set()
    # print(c_table.columns.values)

    # print(get_perfect_idxs(df, 0))
    for i in range (5):
        cur_control = get_z_scores(df,i)[get_control_idxs(df,i)]
        cur_idxs = get_indexes(df,i)[get_control_idxs(df,i)]
        c.update(cur_idxs.tolist())s



            #
            # c_table.loc[i] = [0]*(len(c_table.columns.tolist()))
            # if not (cur_idxs[j] in c_table.columns.values):
            #     c_table[cur_idxs[j]] = []
            # c_table.set_value(cur_idxs[j], i, cur_control[j])
    print(c)


