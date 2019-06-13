import pandas as pd
import numpy as np
from scipy import stats

RESULTS_FILE_ADDR = "Batch_3548123_batch_results.csv"
HITS_NUM =5
MIN_CONTROL_REPET = 3


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


def parse_data(path):
    df = pd.read_csv(path, index_col=0)
    cols = sorted(df.columns.tolist())
    return df[cols]

def get_all_controlers(df):
    controls = pd.DataFrame()
    for i in range(HITS_NUM):
        controls.append([None]*20)
        cur_control = get_scores(df, i)[get_control_idxs(df, i)]
        cur_idxs = get_indexes(df, i)[get_control_idxs(df, i)]
        for j in range(len(cur_idxs)):
            controls.at[i,cur_idxs[j]] = cur_control[j]
    return controls # hits in colomns, sentenses in rows

def get_reaction_time(df,id):
    return df.loc[[id]]["WorkTimeInSeconds"]

def get_average_p(df,id):
    return np.mean(get_z_scores(df,id)[get_perfect_idxs(df,id)])



def control_cor_with_mean(controls,id):
    cur = controls.iloc[[id]]
    p = controls.drop([id]).mean()
    cur = cur.append(p, ignore_index=True)
    cur = cur.dropna(axis = 1).values.tolist()
    r = stats.pearsonr(cur[0],cur[1])
    return r

def clean_controlers_df(controlers, min_repet):
    to_remove = []
    for c in controls:
        if controls[c].dropna().size < min_repet:
            to_remove.append(c)
    for c in to_remove:
        del controls[c]
    return controls

if __name__ == "__main__":
    df = parse_data(RESULTS_FILE_ADDR)
    for i in range(5):
        print(get_reaction_time(df,i))

    controls = get_all_controlers(df)
    controls = clean_controlers_df(controls, MIN_CONTROL_REPET)



    # for i in range(HITS_NUM):
    #     # print(control_cor_with_mean(controls,i))
    #     print("for HIT " + str(i) + " got (pearson-r, p-val) : "+ str(control_cor_with_mean(controls,i)))


    # controls.to_csv("controls.csv") # safe to file

    # controls = pd.melt(controls,value_name="z_score").drop([], axis=1)
    # print(controls)
    # controls = controls.dropna()
    # print(controls)
    # ax1 = controls.plot.scatter(x=controls.columns.tolist(),y=controls.values)






