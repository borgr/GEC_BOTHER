import os
from enum import Enum

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

from GEC_ME_PROJECT.NUCLE.my_NUCLE_parser.AnnotationTypes import AnnotationType
from results_analysis import analyze, plot_mistakes, demo_analyze, get_sentences_mistakes_scores_path, RESULTS_DIR, \
    get_preprocessed_path

### FILES ###
# original MTurk csv file:
from GEC_ME_PROJECT.NUCLE.my_NUCLE_parser.results_analysis import SENTENCES_MISTAKES_SCORE

RESULTS_FILES_ADDR = [os.path.join(RESULTS_DIR, r"Batch_3727145_batch_results.csv"),
                      os.path.join(RESULTS_DIR, r"Batch_4228576_batch_results.csv")]
# results as a melted shape (one sentence per line):
MELTED_DF_ADDR = os.path.join(RESULTS_DIR, r"new_df.csv")
# results as a melted shape with z-scores:
STAND_DF_ADDR = os.path.join(RESULTS_DIR, r"z-scores.csv")
# control sentences address:
CTRL_DF_ADDR = os.path.join(RESULTS_DIR, r"controls_df.csv")

### CONSTANTS:
HITS_NUM = 290
HIT_SIZE = 100
MIN_CONTROL_REPET = 15
DF_SEN_COLS = ["Input.Nucle_ID", "Input.original_text", "Input.is_perfect", "Input.is_control",
               "Answer.mistakeRate"]
DF_HIT_COLS = ["HITId", "WorkerId", "WorkTimeInSeconds", "Input.batch", "MistakeZScore"]
DF_COLS = DF_HIT_COLS + DF_SEN_COLS
# data cleaning:
MIN_WORKING_TIME = 350  # in seconds (=5.83 minutes)
CTRL_CORR_LIM = -0.4  # filtering by correlation threshold
PERFECT_TTEST_ALPHA = 0.05  # the alpha for the t.test of perfect sentences


def standardize_worker(df):
    """
    get a df of a single worker mistake rates and standardize it
    :param df: melted df of a single worker
    :return: melted df of a single worker with z-scores
    """
    # df["MistakeZScore"] = stats.zscore(df["Answer.mistakeRate"]) # older pandas versions
    df = df.assign(MistakeZScore=stats.zscore(df["Answer.mistakeRate"]))
    return df


def standardize_data(df, force=True):
    """
    add z-scores to all workers
    :param df: melted df
    :param force: if True (default) recalculates, otherwise reads from file
    :return: melted df with z-scores
    """
    if not force and os.path.isfile(STAND_DF_ADDR):
        return pd.read_csv(STAND_DF_ADDR)
    df.sort_values('WorkerId')
    workers = set(df["WorkerId"])
    df.set_index(['WorkerId'], inplace=True)
    for worker in workers:
        worker_df = df.loc[worker]
        df.loc[worker] = standardize_worker(worker_df)
    df.reset_index(inplace=True)
    df.to_csv(STAND_DF_ADDR, sep=",", encoding='utf-8', index=False)
    return df


def parse_data(path, force=False):
    """
    get results as the return from MTurk and create melted df (one row for each sentence score)
    :param path: MTurk csv addr or a dataframe concatanating serveral batches
    :return: melted df. saves it as MELTED_DF_ADDR as well
    """
    if not force and os.path.isfile(MELTED_DF_ADDR):
        print("Data already exists, not parsing. Pass force=True to parse_data() from scratch")
        return pd.read_csv(MELTED_DF_ADDR)
    try:
        orig_df = pd.read_csv(path)
    except ValueError:
        orig_df = path
    new_df = []
    print("Parsing Mturk batches")
    for i in tqdm(list(range(len(orig_df)))):
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
            # new_df.loc[(i * HIT_SIZE) + j] = new_row
            new_df.append(new_row)
    new_df = pd.DataFrame(new_df, columns=DF_COLS)
    new_df.to_csv(MELTED_DF_ADDR, sep=",", encoding='utf-8', index=False)
    return new_df


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
            averages[sen] = (np.mean(scores), len(scores))  # tuple: average and number of annotators
    sentences = sentences - black_list
    controls_df = df.loc[df["Input.Nucle_ID"].isin(sentences)]
    excluded_average = []  # not including this notation
    for index, row in controls_df.iterrows():
        ave = averages[row['Input.Nucle_ID']]
        excluded_average.append(calc_excluded_average(row["MistakeZScore"], ave[0], ave[1]))
    controls_df.insert(2, "ExcludedAverage", excluded_average)
    # controls_df.to_csv("controls_df.csv", sep=",", encoding='utf-8')
    return controls_df


def calc_excluded_average(score, average, n):
    """
    calculate control sentence excluded average
    :param score: score to exclude
    :param average: average include this score
    :param n: number of scores per sentence
    :return: excluded average
    """
    return ((average * n) - score) / (n - 1)


def get_perfects_mean(worker_df):
    """
    return perfect sentences mean
    :param worker_df: melted df of a single worker
    :return: perfect sentences mean
    """
    return np.mean(worker_df.loc[worker_df['Input.is_perfect']]["MistakeZScore"])


def filter_by_time(df, time_lim):
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
    """
    print some stats about the workers. for cleaning data purposes
    :param df: melted df
    """
    all_times = []
    zscores = []
    df = df.copy()
    workers = set(df["WorkerId"])
    df.set_index(['WorkerId'], inplace=True)
    for worker in workers:
        worker_df = df.loc[worker]
        pz_score = get_perfects_mean(worker_df)
        times = set(worker_df["WorkTimeInSeconds"])
        times = [(time / 60) for time in times]
        all_times = all_times + times
        zscores.append(pz_score)
        # if len(times) >1:
        #     print("hitter: ", worker)
        #     print("number of sentences per hitter: ",worker_df.shape[0])
        #     print("perfect sentences mean z-score: ", pz_score)
        #     print("working times (in minutes): ", times, "mean time is: ", np.mean(times))
        #     print()
    # print(sorted(all_times))
    plt.scatter(x=list(range(290)), y=sorted(all_times)[:290])
    plt.title("HITS sorted by working time")
    plt.xlabel("HIT number")
    plt.ylabel("working time (minutes)")
    # print("num of positives: ", len([i for i in zscores if i>0]))
    # plt.scatter(x=list(range(len(workers))),y=sorted(zscores))
    plt.show()


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
        x = stats.ttest_ind(non_perfect, perfects, equal_var=True)
        if (x[1] < (alpha * 2)) and np.mean(perfects) > 0:
            black_list.add(worker)
    df = df.loc[df['WorkerId'].isin(workers - black_list)]
    return df, black_list


def filter_by_corr(df, controls_df, cor_lim):
    """
    filter workers whose control sentences scores were significantly uncorrelated with the rest of the workers
    :param df: mekted df
    :param controls_df: control sentences
    :param cor_lim: workers with lower correlation then that will be eliminated.
    :return:
    """
    black_list = set()
    workers = set(controls_df["WorkerId"])
    corrs = []
    all_corrs = []
    for worker in workers:
        worker_df = controls_df.loc[controls_df['WorkerId'] == worker]
        worker_scores = worker_df["MistakeZScore"]
        mean_scores = worker_df["ExcludedAverage"]
        cor = stats.pearsonr(worker_scores, mean_scores)
        all_corrs.append(cor[0])
        if cor[0] < cor_lim:
            black_list.add(worker)
        else:
            corrs.append(cor[0])
    # plt.scatter(list(range(len(all_corrs))),sorted(all_corrs))
    # plt.title('Control sentences score correlations')
    # plt.xlabel('worker')
    # plt.ylabel('pearson r')
    # plt.show()
    df = df.loc[df['WorkerId'].isin(workers - black_list)]
    # print("negs:")
    # print(len(corrs),len([i for i in corrs if i < 0]))

    return df, black_list


def plot_cor_sentences(control_df):
    """
    plot a graph of the different control sentences scores
    :param control_df: control sentences Data Frame
    :return: None
    """
    plot_ctrol_df = pd.DataFrame([control_df['Input.Nucle_ID'], control_df['MistakeZScore']]).transpose()
    means = dict()
    plot_ctrol_df["mean"] = 0
    for id in set(plot_ctrol_df['Input.Nucle_ID']):
        means[id] = np.mean(plot_ctrol_df.loc[plot_ctrol_df['Input.Nucle_ID'] == id, "MistakeZScore"])
        plot_ctrol_df.loc[plot_ctrol_df['Input.Nucle_ID'] == id, "mean"] = means[id]
        # plot_ctrol_df.loc["mean"][plot_ctrol_df['Input.Nucle_ID'] == id] = means[id] # old pandas version
    plot_ctrol_df.sort_values(by=['mean'], inplace=True)
    i = 0
    b = set()
    x = list(plot_ctrol_df['Input.Nucle_ID'])
    for id in x:
        if id not in b:
            b.add(id)
            plot_ctrol_df.loc[plot_ctrol_df['Input.Nucle_ID'] == id, 'Input.Nucle_ID'] = -i
            i += 1
    plot_ctrol_df["Input.Nucle_ID"] = -plot_ctrol_df["Input.Nucle_ID"]
    plt.title('NUCLE sentences different scores')
    plt.xlabel('control sentence')
    plt.ylabel('z-score')
    plt.scatter(plot_ctrol_df['Input.Nucle_ID'], plot_ctrol_df['MistakeZScore'], color='skyblue')
    plt.scatter(plot_ctrol_df['Input.Nucle_ID'], plot_ctrol_df['mean'], color='maroon')
    # plt.scatter(range(len(set(control_df['Input.Nucle_ID']))),means, color='maroon')
    plt.show()
    print("mean:", np.mean(list((plot_ctrol_df['mean']))))
    print("Consider changing the correlation of filtering according to this")


def clean_data(df):
    """
    clean MTurk workers from the data by time, perfect sentences score and control sentences score.
    :param df: standardized melted df
    :return: the df without the lines that were assessed by the declined workers
    """
    print("df len before: ", df.shape[0])
    df, b = filter_by_time(df, MIN_WORKING_TIME)
    print("df len after time: ", df.shape[0], ", annotators cleaned by time: ", len(b))
    df, b = filter_by_perfect(df, PERFECT_TTEST_ALPHA)
    print("df len after perfect: ", df.shape[0], ", annotators cleaned by perfect: ", len(b))
    control_df = parse_control_sentences(df)
    df, b = filter_by_corr(df, control_df, CTRL_CORR_LIM)
    plot_cor_sentences(control_df)
    print("df len after control: ", df.shape[0], ", annotators cleaned by control: ", len(b))
    df.to_csv(get_preprocessed_path(tpe), sep=",", encoding='utf-8', index=False)
    return df


if __name__ == "__main__":
    force = False
    tpe = AnnotationType.SERCL
    # tpe = AnnotationType.COLLAPSED_ERRANT
    # tpe = AnnotationType.ERRANT
    # tpe = AnnotationType.NUCLE
    for tpe in iter(AnnotationType):
        error = "std"
        if not force and os.path.isfile(get_preprocessed_path(tpe)):
            print("Skipping preprocessing, already done")
            clean_df = pd.read_csv(get_preprocessed_path(tpe))
        else:
            print("reading from", [os.path.abspath(os.path.normpath(fl)) for fl in RESULTS_FILES_ADDR])
            dfs = [pd.read_csv(os.path.normpath(path)) for path in RESULTS_FILES_ADDR]
            df = pd.concat(dfs).reset_index()
            # force = True
            melted = parse_data(df, force)
            workers_stats(melted)
            zdf = standardize_data(melted, force)
            clean_df = clean_data(zdf)
        # mistakes = pd.read_csv(r" GEC_Project\GEC_ME_PROJECT-master\GEC_ME_PROJECT\NUCLE\my_NUCLE_parser\debug.csv", index_col=0)
        # plot_mistakes(mistakes)
        force = False
        force_bootstrap = True
        analyze(clean_df, tpe, error, force=force, force_bootstrap=force_bootstrap)
        # demo_analyze()
