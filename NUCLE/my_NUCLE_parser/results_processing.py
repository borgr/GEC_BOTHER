import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from results_analysis import analize, plot_mistakes, demo_analize


RESULTS_FILE_ADDR = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\DA\results\Batch_3727145_batch_results .csv"
NEW_DF_ADDR = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\DA\results\new_df.csv"
STAND_DF_ADDR = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\DA\results\z-scores.csv"
CTRL_DF_ADDR = r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\DA\results\controls_df.csv"
HITS_NUM = 290
HIT_SIZE = 100
MIN_CONTROL_REPET = 15

DF_SEN_COLS = ["Input.Nucle_ID", "Input.original_text", "Input.is_perfect", "Input.is_control",
               "Answer.mistakeRate"]
DF_HIT_COLS = ["HITId", "WorkerId", "WorkTimeInSeconds", "Input.batch", "MistakeZScore"]
DF_COLS = DF_HIT_COLS + DF_SEN_COLS


# gets an HIT id and return an array of size 100 with the indexes of the sentences in the hit
def get_indexes(df, HIT_id):
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
def get_scores(df, HIT_id):
    return np.array([df[col][HIT_id] for col in df if col.startswith('Answer.mistakeRate')])


# return an array of size 100 which contains this HIT z-scores
def get_z_scores(df, HIT_id):
    return stats.zscore(get_scores(df, HIT_id))


# return an array of size 15 which contains this HIT perfect sentences scores
def HIT_p_scores(df, HIT_id):
    return get_z_scores(df, HIT_id)[get_perfect_idxs(df, HIT_id)]



# returns the average z-score of perfect sentences
def HIT_p_average(df, HIT_id):
    return np.mean(HIT_p_scores(df, HIT_id))


def parse_data(path):
    df = pd.read_csv(path)
    cols = sorted(df.columns.tolist())
    return df[cols]


def get_all_controlers(df):
    controls = pd.DataFrame()
    for i in range(HITS_NUM):
        controls.append([None] * 20)
        cur_control = get_scores(df, i)[get_control_idxs(df, i)]
        cur_idxs = get_indexes(df, i)[get_control_idxs(df, i)]
        for j in range(len(cur_idxs)):
            controls.at[i, cur_idxs[j]] = cur_control[j]
    return controls  # hits in colomns, sentenses in rows


def get_reaction_time(df, id):
    return df.iloc[id]["WorkTimeInSeconds"]


def get_average_p(df, id):
    return np.mean(get_z_scores(df, id)[get_perfect_idxs(df, id)])


def control_cor_with_mean(controls, id):
    cur = controls.iloc[[id]]
    p = controls.drop([id]).mean()
    cur = cur.append(p, ignore_index=True)
    cur = cur.dropna(axis=1).values.tolist()
    r = stats.pearsonr(cur[0], cur[1])
    return r


def clean_controlers_df(controls, min_repet):
    to_remove = []
    for c in controls:
        if controls[c].dropna().size < min_repet:
            to_remove.append(c)
    for c in to_remove:
        del controls[c]
    return controls


def print_hitter_stats(df, controls, i):
    print("Hitter number " + str(i) + ":")
    print("Work time in minutes: " + str(get_reaction_time(df, i) / 60))
    print("corolation amoung control sentences (pearson-r, p-val): " + str(
        (control_cor_with_mean(controls, i))))
    print("average z-score on perfect sentences: " + str(get_average_p(df, i)))
    print()
    times.append(get_reaction_time(df, i) / 60)


def get_all_z_scores(results_path):
    df = parse_data(results_path)
    z_scores = pd.DataFrame(
        columns=[col for col in df if (col.startswith('Answer.mistakeRate') or col.startswith("A"))])
    for i in range(HITS_NUM):
        z_scores.loc[i] = get_z_scores(df, i)
    return z_scores


def standardize_worker(df):
    df["MistakeZScore"] = stats.zscore(df["Answer.mistakeRate"])
    return df



def standardize_data(df):
    df.sort_values('WorkerId')
    workers = set(df["WorkerId"])
    df.set_index(['WorkerId'], inplace=True)
    for worker in workers:
        worker_df = df.loc[worker]
        x = standardize_worker(worker_df)
        df.loc[worker] = x
    df.to_csv("z-scores.csv", sep=",", encoding='utf-8')


def parse_data2(path):
    orig_df = pd.read_csv(path)
    new_df = pd.DataFrame(columns=DF_COLS)
    for i in tqdm(range(HITS_NUM)):
        for j in range(1, HIT_SIZE + 1):
            new_row = []
            for col in DF_COLS:
                if col == "MistakeZScore":
                    new_row.append(0)  # a temp place holder
                    continue
                if col in DF_SEN_COLS:
                    col = col + str(j)
                x = orig_df[col]
                new_row.append(x.iloc[i])
            new_df.loc[(i * HIT_SIZE) + j] = new_row
    new_df.to_csv("new_df.csv", sep=",", encoding='utf-8')


def parse_control_sentences(df):
    """
    parse control sentences - remove  ones with less then MIN_CONTROL_REPET occurences and calc excluded average per each row
    :param df: sentences with z-scores
    :return: df of just control sentences with excluded average for each row
    """
    controls_df = df.loc[df['Input.is_control'] == True]
    sentences = set(controls_df["Input.Nucle_ID"])
    averages = dict()
    black_list = set()
    for sen in sentences:
        scores = df.loc[df['Input.Nucle_ID'] == sen]["MistakeZScore"]
        if len(scores) < MIN_CONTROL_REPET:
            black_list.add(sen)
        else:
            averages[sen] = (np.mean(scores), len(scores)) #tuple: average and number of annotators
    sentences = sentences - black_list
    controls_df = df.loc[df["Input.Nucle_ID"].isin(sentences)]
    excluded_average = []  # not including this notation
    for index, row in controls_df.iterrows():
        ave = averages[row['Input.Nucle_ID']]
        excluded_average.append(calc_excluded_average(row["MistakeZScore"], ave[0],ave[1]))
    controls_df.insert(2, "ExcludedAverage", excluded_average)
    controls_df.to_csv("controls_df.csv", sep=",", encoding='utf-8')
    return controls_df

def calc_excluded_average(score, average, n):
    return ((average*n)-score)/(n-1)

def get_perfects_mean(worker_df):
    return np.mean(worker_df.loc[worker_df['Input.is_perfect']]["MistakeZScore"])


def filter_by_time(df,time_lim):
    """
    filter data by the worker working time - if worker minimum time is lower than time limit it will be removed
    :param df: sentences data frame
    :param time_lim: time limit in seconds. faster workers will be removed
    :return: the filtered DF
    """
    black_list = set()
    workers = set(df["WorkerId"])
    for worker in workers:
        worker_df = df.loc[df['WorkerId'] == worker]
        times = set(worker_df["WorkTimeInSeconds"])
        if np.min([int(i) for i in times]) < time_lim:
            black_list.add(worker)
    df = df.loc[df['WorkerId'].isin(workers - black_list)]
    return df, black_list


def workers_stats(df):
    all_times = []
    zscores = []
    workers = set(df["WorkerId"])
    df.set_index(['WorkerId'], inplace=True)
    for worker in workers:
        worker_df = df.loc[worker]
        pz_score = get_perfects_mean(worker_df)
        times = set(worker_df["WorkTimeInSeconds"])
        times = [(time/60) for time in times]
        all_times = all_times + times
        zscores.append(pz_score)
        # if len(times) >1:
        #     print("hitter: ", worker)
        #     print("number of sentences per hitter: ",worker_df.shape[0])
        #     print("perfect sentences mean z-score: ", pz_score)
        #     print("working times (in minutes): ", times, "mean time is: ", np.mean(times))
        #     print()
    # print(sorted(all_times))
    # plt.scatter(x=list(range(50)),y=sorted(all_times)[:50])
    # print("num of positives: ", len([i for i in zscores if i>0]))
    # plt.scatter(x=list(range(len(workers))),y=sorted(zscores))
    # plt.show()



def filter_by_perfect(df, alpha):
    """
    filter workers whose perfect sentences score was statistically significantly higher then the ones with mistakes
    :param df: sentences data frame
    :param alpha: single sided t.test alpha
    :return: a tuple of the edited df and the black list of workers
    """
    black_list = set()
    workers = set(df["WorkerId"])
    for worker in workers:
        worker_df = df.loc[df['WorkerId'] == worker]
        perfects = worker_df.loc[worker_df['Input.is_perfect'] == True]["MistakeZScore"]
        non_perfect = worker_df.loc[worker_df['Input.is_perfect'] == False]["MistakeZScore"]
        x=  stats.ttest_ind(non_perfect,perfects, equal_var=True)
        if (x[1] < (alpha*2)) and np.mean(perfects)>0:
            black_list.add(worker)
    df = df.loc[df['WorkerId'].isin(workers - black_list)]
    return df, black_list

def filter_by_corr(df,controls_df, cor_lim):
    black_list = set()
    workers = set(controls_df["WorkerId"])
    corrs = []
    for worker in workers:
        worker_df = controls_df.loc[controls_df['WorkerId'] == worker]
        worker_scores = worker_df["MistakeZScore"]
        mean_scores = worker_df["ExcludedAverage"]
        cor = stats.pearsonr(worker_scores, mean_scores)
        if (cor[0] < cor_lim):
            black_list.add(worker)
        else: corrs.append(cor[0])
    df = df.loc[df['WorkerId'].isin(workers - black_list)]
    print("negs:")
    print(len(corrs),len([i for i in corrs if i < 0]))
    # plt.scatter(list(range(len(corrs))),sorted(corrs))
    # plt.xlabel('worker', fontsize=18)
    # plt.ylabel('pearson r', fontsize=16)
    # plt.show()
    return df, black_list

def get_controls_sd(controls_df):
    sentences = set(controls_df["Input.Nucle_ID"])
    stds = []
    for sen in sentences:
        scores = df.loc[df['Input.Nucle_ID'] == sen]["MistakeZScore"]
        stds.append(np.std(scores))
    return stds


def clean_data(df):
    df = filter_by_time(df,350)[0]
    df = filter_by_perfect(df,0.05)[0]
    control_df = parse_control_sentences(df)
    df, b = filter_by_corr(df, control_df,-0.4)
    # plt.xlabel('sentence ID', fontsize=18)
    # plt.ylabel('z-score', fontsize=16)
    # plt.scatter(control_df['Input.Nucle_ID'], control_df['MistakeZScore'])
    # plt.show()
    return df


if __name__ == "__main__":
    df = pd.read_csv(STAND_DF_ADDR)
    df = clean_data(df)

    print(df.shape)

    # mistakes = pd.read_csv(r"C:\Users\ofir\Documents\University\year2\GEC Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\my_NUCLE_parser\debug.csv", index_col=0)
    # plot_mistakes(mistakes)
    # analize(df)
    # demo_analize()






