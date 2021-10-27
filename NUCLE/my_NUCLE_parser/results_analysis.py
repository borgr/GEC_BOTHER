import os

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from GEC_ME_PROJECT.NUCLE.my_NUCLE_parser.AnnotationTypes import AnnotationType
from my_parser import NUCLE_DB_ADDR, parse_sentence, MISTAKES_INX

### FILES ###
RESULTS_DIR = r"DA/results/"
FIG_DIR = os.path.join(RESULTS_DIR, "graphs")
# RESULTS_FILE_ADDR = os.path.join(RESULTS_DIR, r"Batch_3727145_batch_results.csv")
TO_MTURK_FILE_PATH = r"NUCLE/toMturk/mTurk_csv.csv"
SENTENCES_MISTAKES_SCORE = os.path.join(RESULTS_DIR, r"sentences_mistakes_scores.csv")
ERRANT_SENTENCES_MISTAKES_SCORE = os.path.join(RESULTS_DIR, r"sentences_mistakes_scores_errant.csv")
SERCL_SENTENCES_MISTAKES_SCORE = os.path.join(RESULTS_DIR, r"sentences_mistakes_scores_sercl.csv")
ERRANT_PREPROCESSED = os.path.join(RESULTS_DIR, r"preprocessed_errant.csv")
PREPROCESSED = os.path.join(RESULTS_DIR, r"preprocessed.csv")
SERCL_PREPROCESSED = os.path.join(RESULTS_DIR, r"preprocessed_sercl.csv")
ERRANT_DB_ADDR = r"errant-master/out_auto_m2"
SERCL_DB_ADDR = r"errant-master/sercl_auto_m2"
CONFIDENCE = 0.9

# features:
SERCL_MISTAKE_TYPES = ['NOUN->PROPN', 'NOUN->None', 'ADJ->ADV', 'ADP->SCONJ', 'PUNCT->None', 'DET->DET', 'PRON->DET',
                       'ADP->VERB', 'None->PRON', 'ADV->ADV', 'ADP->None', 'PRON->PRON', 'VERB->None', 'NUM->None',
                       'ADV->None', 'ADJ->ADJ', 'ADP->DET', 'VERB->ADJ', 'CCONJ->CCONJ', 'NOUN->ADJ', 'VERB->ADP',
                       'ADJ->None', 'DET->None', 'None->CCONJ', 'ADJ->DET', 'DET->NOUN', 'PROPN->None', 'None->ADV',
                       'SCONJ->ADP', 'None->AUX', 'DET->PRON', 'NOUN->PRON', 'PART->None', 'None->VERB', 'None->DET',
                       'ADJ->VERB', 'VERB->AUX', 'NOUN->NOUN', 'None->SCONJ', 'None->NOUN', 'NOUN->VERB',
                       'PUNCT->CCONJ', 'DET->ADJ', 'None->PUNCT', 'None->ADJ', 'AUX->None', 'None->PART', 'ADV->VERB',
                       'PROPN->NOUN', 'AUX->AUX', 'PART->ADP', 'ADP->PART', 'PRON->NOUN', 'ADV->ADP', 'ADJ->NOUN',
                       'CCONJ->None', 'ADP->ADP', 'AUX->VERB', 'SCONJ->None', 'VERB->NOUN', 'NOUN->DET', 'PROPN->PROPN',
                       'ADV->ADJ', 'ADP->ADV', 'CCONJ->PUNCT', 'VERB->VERB', 'None->ADP', 'PRON->None', 'PUNCT->PUNCT',
                       'SCONJ->SCONJ']
NUCLE_MISTAKE_TYPES = ["Vt", "Vm", "V0", "Vform", "SVA", "ArtOrDet", "Nn", "Npos", "Pform", "Pref", "Prep",
                       "Wci",
                       "Wa", "Wform", "Wtone", "Srun", "Smod", "Spar", "Sfrag", "Ssub", "WOinc", "WOadv",
                       "Trans",
                       "Mec", "Rloc-", "Cit", "Others", "Um"]
COARSE_ERRANT_TYPES = ['VERB:FORM', 'VERB:TENSE', 'NOUN', 'PREP', 'PRON', 'SPELL',
                       'ADV', 'NOUN:POSS', 'ORTH', 'DET', 'OTHER', 'ADJ', 'CONJ', 'NOUN:NUM',
                       'CONTR', 'PUNCT', 'PART', 'NOUN:INFL', 'VERB:SVA', 'WO', 'VERB',
                       'VERB:INFL', 'UNK', 'MORPH', 'ADJ:FORM']
ERRANT_MISTAKE_TYPES = ['R:VERB:FORM', 'R:VERB:TENSE', 'M:NOUN', 'U:PREP', 'M:VERB:TENSE', 'R:PRON',
                        'R:SPELL',
                        'R:ADV', 'U:NOUN:POSS', 'R:ORTH', 'M:DET', 'U:OTHER', 'R:NOUN', 'U:ADV', 'M:ADJ',
                        'R:PREP',
                        'R:CONJ', 'U:NOUN', 'M:VERB:FORM', 'R:NOUN:NUM', 'R:NOUN:POSS', 'U:VERB:FORM',
                        'R:CONTR',
                        'M:ADV', 'U:DET', 'U:PUNCT', 'M:PRON', 'U:PART', 'R:NOUN:INFL', 'M:PART', 'M:CONJ',
                        'R:VERB:SVA',
                        'M:PUNCT', 'R:PUNCT', 'M:NOUN:POSS', 'U:VERB:TENSE', 'U:PRON', 'U:CONJ', 'R:WO',
                        'R:VERB',
                        'R:ADJ', 'M:PREP', 'R:VERB:INFL', 'U:ADJ', 'U:VERB', 'M:OTHER', 'U:CONTR', 'UNK',
                        'R:OTHER',
                        'R:PART', 'R:MORPH', 'R:DET', 'M:VERB', 'R:ADJ:FORM']

# EVALUATION_FEATURES = ["Nucle_ID"] + NUCLE_MISTAKE_TYPES + ["TotalMistakes", "z-score"]
NUM_OF_BOOTSTRAP_ITER = 10000  # 10000


def get_evaluation_features(tpe):
    return ["Nucle_ID"] + get_mistake_types(tpe) + ["TotalMistakes", "z-score"]


def get_mistake_types(tpe: AnnotationType):
    if tpe == AnnotationType.ERRANT:
        types = ERRANT_MISTAKE_TYPES
    elif tpe == AnnotationType.COLLAPSED_ERRANT:
        types = COARSE_ERRANT_TYPES
    elif tpe == AnnotationType.NUCLE:
        types = NUCLE_MISTAKE_TYPES
    elif tpe == AnnotationType.SERCL:
        types = SERCL_MISTAKE_TYPES
    else:
        raise ValueError(f"unknown type: {tpe}")
    return types


def get_sentences_mistakes_scores_path(tpe: AnnotationType):
    if tpe in (AnnotationType.ERRANT, AnnotationType.COLLAPSED_ERRANT):
        return ERRANT_SENTENCES_MISTAKES_SCORE
    elif tpe == AnnotationType.NUCLE:
        return SENTENCES_MISTAKES_SCORE
    elif tpe == AnnotationType.SERCL:
        return SERCL_SENTENCES_MISTAKES_SCORE
    else:
        raise ValueError(f"unknown type {tpe}")


def get_preprocessed_path(tpe: AnnotationType):
    if tpe in [AnnotationType.ERRANT, AnnotationType.COLLAPSED_ERRANT]:
        return ERRANT_PREPROCESSED
    elif tpe == AnnotationType.NUCLE:
        return PREPROCESSED
    elif tpe == AnnotationType.SERCL:
        return SERCL_PREPROCESSED
    raise ValueError(f"type unknown:{tpe}")


def nucle_to_errant_id_dict(nucle_addr, errant_addr):
    """
    create a dictionary of id's  nucle of form- original line : errant line
    :param nucle_addr: original NUCLE m2 file addr
    :param errant_addr: ERRANT m2 file addr
    :return:
    """
    with open(nucle_addr) as fl:
        nucle_lines = fl.read().splitlines()
    with open(errant_addr, encoding="ISO-8859-2") as fl:
        errant_lines = fl.read().splitlines()
    n_counter, e_counter, n_len = 0, 0, len(nucle_lines)
    d = dict()
    while True:
        if n_counter >= n_len:
            break
        if nucle_lines[n_counter].startswith("S"):
            while not errant_lines[e_counter].startswith("S"):
                e_counter += 1
            d[n_counter] = e_counter
            e_counter += 1
        n_counter += 1
    return d


def get_ERRANT_atributes():
    """
    :return: all posible mistakes by errant
    """
    errant_lines = open(ERRANT_DB_ADDR).read().splitlines()
    a = set()
    for id, line in enumerate(errant_lines):
        if line.startswith("S"):
            mistakes = parse_sentence(errant_lines, id)[MISTAKES_INX]
            for mistake in mistakes:
                a.add(mistake[2])
    return a


def get_senteces_by_df(nucle_addr, df, tpe: AnnotationType, force=True):
    """
    :param nucle_addr: original m2 NUCLE file
    :param df: melted filtered df
    :param tpe: type of annotation
    :return: df in which every sentence is a vector of mistakes with z-score
    """
    if not force and os.path.isfile(get_sentences_mistakes_scores_path(tpe)):
        res = pd.read_csv(get_sentences_mistakes_scores_path(tpe))
        return res
    with open(nucle_addr, encoding="ISO-8859-2") as fl:
        lines = fl.read().splitlines()
    # sentences = pd.DataFrame(columns=EVALUATION_FEATURES)
    # d = nucle_to_errant_id_dict(NUCLE_DB_ADDR, ERRANT_DB_ADDR)
    print("Combining sentences their mistakes and their annotators z-score")
    total = []
    zscore = []
    nucle_id = []
    mistake_types = get_mistake_types(tpe)
    mistakes_counts = {name: [] for name in mistake_types}
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # sentences.loc[index] = [0] * len(EVALUATION_FEATURES)
        id = int(row["Input.Nucle_ID"]) - 1
        # if errant:
        #     id = d[id]
        mistakes = parse_sentence(lines, id)[MISTAKES_INX]
        nucle_id.append(id)
        total.append(0)
        for mistake_name in mistake_types:
            mistakes_counts[mistake_name].append(0)
        for mistake in mistakes:
            if mistake[2] not in mistakes_counts:
                if tpe != AnnotationType.SERCL:
                    raise ValueError(f"unknown mistake {mistake[2]}")
            else:
                mistakes_counts[mistake[2]][index] += 1
            total[index] += 1
        zscore.append(df.loc[index]["MistakeZScore"])
    data = [nucle_id] + [mistakes_counts[mistake_name] for mistake_name in mistake_types] + [total, zscore]
    sentences = pd.DataFrame(data=np.array(data).transpose(), columns=get_evaluation_features(tpe))

    # sentences.loc[index]["Nucle_ID"] = id
    # for mistake in mistakes:
    #     sentences.loc[index][mistake[2]] += 1
    #     sentences.loc[index]["TotalMistakes"] += 1
    # sentences.loc[index]["z-score"] = df.loc[index]["MistakeZScore"]
    sentences.to_csv(get_sentences_mistakes_scores_path(tpe), sep=",", encoding='utf-8', index=False)
    return sentences


def mistakes_stats(sentences):
    """
    create a df of statistics of the different mistakes
    :param sentences: df in which every sentence is a vector of mistakes with z-score
    :return: df of statistics of the different mistakes
    """
    sentences = get_X(sentences)
    df = pd.DataFrame(
        columns=["number_of_sentences", "perc_of_sentences", "total_appearance", "perc_of_appearance"])
    df["number_of_sentences"] = sentences.astype(bool).sum(axis=0)
    df["total_appearance"] = sentences.sum(axis=0)
    df["perc_of_sentences"] = df["number_of_sentences"] / sentences.shape[0]
    df["perc_of_appearance"] = df["total_appearance"] / df["total_appearance"]["TotalMistakes"]
    return df


def get_X(sentences):
    """
    get the domain for the regression - clean irrelevant columns from the df
    :param sentences: senteces df
    :return: same df without irrelevant columns
    """
    return sentences.drop('z-score', 1).drop('Nucle_ID', 1)


def get_weights(sentences, reg_flag):
    """
    perform linear regression and return the weights
    :param sentences: df of sentences as a vector of mistakes and a z-score
    :param reg_flag: if True - include number of mistakes in the regression, else - don't include it
    :return: regression weights a a vector.
    """
    y = np.array(sentences["z-score"])
    if reg_flag:
        X = np.array(get_X(sentences))
        bias = 89138352537  # regression bug magic solution, all results had this bias, may change on different computer
    else:
        X = np.array(get_X(sentences).drop('TotalMistakes', 1))
        bias = 0
    reg = LinearRegression().fit(X, y)
    return reg.coef_ - bias


def bootstrap_weights(sentences, tpe: AnnotationType, force=True):
    """
    resample and estimate data values NUM_OF_BOOTSTRAP_ITER times.
    :param sentences: df of sentences as a vector of mistakes and a z-score
    :param tpe: type of annotations
    :param force: if true recalculates even if results can be loaded
    :return:
    mistakes - a df of mistake stats
    mistakes.transpose()["upper"] - a vector of top 97.5% scores for the different mistake types
    mistakes.transpose()["lower"] - a vector of bottom 2.5% scores for the different mistake types
    ranks - a df of mistake's ranks over the iterations
    """
    if tpe == AnnotationType.ERRANT:
        name = "ERRANT"
    elif tpe == AnnotationType.COLLAPSED_ERRANT:
        name = "fine_ERRANT"
    elif tpe == AnnotationType.NUCLE:
        name = "NUCLE"
    elif tpe == AnnotationType.SERCL:
        name = "SERCL"
    else:
        raise ValueError(f"unknow type: {AnnotationType}")
    mistakes_path = os.path.join(RESULTS_DIR, f"cache_bootstrap_{name}.csv")
    ranks_path = os.path.join(RESULTS_DIR, f"ranks_{name}.csv")
    if os.path.isfile(mistakes_path) and os.path.isfile(ranks_path) and not force:
        return pd.read_csv(mistakes_path, index_col=0), pd.read_csv(ranks_path, index_col=0)

    types = get_mistake_types(tpe)
    mistakes = []
    ranks = []
    for i in tqdm(list(range(NUM_OF_BOOTSTRAP_ITER))):
        sample = sentences.sample(n=sentences.shape[0], replace=True)
        weights = list(get_weights(sample, False))
        mistakes.append(weights)
        # arg_sort
        rank_row = [0] * len(weights)
        for j, x in enumerate(sorted(range(len(weights)), key=lambda y: weights[y])):
            rank_row[x] = j
        ranks.append(rank_row)
    mistakes = pd.DataFrame(mistakes, columns=types)
    ranks = pd.DataFrame(ranks, columns=types)
    uppers = []
    upper_ranks = []
    lowers = []
    lower_ranks = []
    k = int(NUM_OF_BOOTSTRAP_ITER * (1 - CONFIDENCE) / 2)
    # print(f"considering {k} from top out of {len(mistakes)} == {NUM_OF_BOOTSTRAP_ITER}")
    # print(f"Changing the confidence percentage doesn't change importance... why?")
    for mis in mistakes:
        uppers.append(np.partition(mistakes[mis], -k)[-k])
        lowers.append(np.partition(mistakes[mis], -(NUM_OF_BOOTSTRAP_ITER - k))[-(NUM_OF_BOOTSTRAP_ITER - k)])
        upper_ranks.append(np.partition(ranks[mis], -k)[-k])
        lower_ranks.append(
            np.partition(ranks[mis], -(NUM_OF_BOOTSTRAP_ITER - k))[-(NUM_OF_BOOTSTRAP_ITER - k)])

    mistakes.loc["upper"] = uppers
    mistakes.loc["lower"] = lowers
    mistakes.loc["lower_ranks"] = lower_ranks
    mistakes.loc["upper_ranks"] = upper_ranks
    mistakes.loc["mean_rank"] = [np.mean(ranks[mt]) for mt in ranks]
    mistakes.to_csv(mistakes_path, sep=",", encoding='utf-8', index=True)  # save for cache
    ranks.to_csv(ranks_path, sep=",", encoding='utf-8', index=True)  # save for cache
    return mistakes, ranks


def get_saving_name(tpe):
    if tpe == AnnotationType.ERRANT:
        type_name = "ERRANT"
        full_type_name = "coarsegrained " + type_name
    elif tpe == AnnotationType.COLLAPSED_ERRANT:
        type_name = "ERRANT"
        full_type_name = "finegrained " + type_name
    elif tpe == AnnotationType.NUCLE:
        type_name = "NUCLE"
        full_type_name = type_name
    elif tpe == AnnotationType.SERCL:
        type_name = "SERCL"
        full_type_name = type_name
    else:
        raise ValueError(f"unknwon type {tpe}")
    return type_name, full_type_name


def plot_mistakes(all_mistakes, mistakes, ranks, tpe):
    """
    plot graph of mistakes scores
    :param all_mistakes: a df with all bootstrap repeats scores
    :param mistakes: mistakes statistics
    :param ranks: mistake ranks of the different bootstrapping iterations df
    :param tpe: type of annotations
    :return: None. plots 3 graphs.
    """
    types = get_mistake_types(tpe)
    type_name, full_type_name = get_saving_name(tpe)

    melted_mistakes = pd.melt(all_mistakes)
    melted_mistakes["weights"] = 0
    for mistake in types:
        # if mistake != "TotalMistakes":
        val = mistakes["weights"][mistake]
        melted_mistakes["weights"][melted_mistakes["variable"] == mistake] = val

    # graph 1 - boxplots:
    melted_mistakes.sort_values(by=['weights'], inplace=True)
    sns.set(style="whitegrid")
    ax = sns.boxplot(x="variable", y="value", data=melted_mistakes)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(xlabel='Error categories', ylabel='linear regression weight')
    # plt.title(f'{type_name} error categories sorted by "bothering" boxplots')
    plt.show()

    # graph 2 - bars:
    mistakes.sort_values(by=['weights'], inplace=True)
    mistakes.drop('TotalMistakes', inplace=True)
    # x = [mis for mis in mistakes.index if mis != "TotalMistakes"] #sorted labels
    means = np.array(mistakes["weights"])
    conf = np.array([[mistakes["lower"][mis], mistakes["upper"][mis]] for mis in mistakes.index])
    yerr = np.c_[means - conf[:, 0], conf[:, 1] - means].T
    plt.bar(mistakes.index, means, yerr=yerr, color="lightseagreen")
    plt.xlabel('Error categories')
    plt.ylabel('Linear regression weight')
    plt.title('{ftype_name} error categories sorted by "bothering" with estimated 95% CI based on boostrapping')
    plt.xticks(rotation=90)
    save_to = os.path.join(FIG_DIR, f"{CONFIDENCE}_{full_type_name}_weights.png")
    plt.tight_layout()
    plt.savefig(save_to)
    print(f"saving to {os.path.abspath(save_to)}")
    plt.show()

    # graph 3 - ranks:
    mistakes.sort_values(by=['mean_rank'], inplace=True)
    # x = [mis for mis in mistakes.index if mis != "TotalMistakes"] #sorted labels
    means = np.array(mistakes["mean_rank"])
    conf = np.array([[mistakes["lower_rank"][mis], mistakes["upper_rank"][mis]] for mis in mistakes.index])
    yerr = np.c_[means - conf[:, 0], conf[:, 1] - means].T
    plt.bar(mistakes.index, means, yerr=yerr, color="darksalmon")
    plt.xlabel('Error categories')
    plt.ylabel("Bothering - rank")
    # plt.title(
    #     f'{type_name} error categories average rank (with {CONFIDENCE}% CI) based on bootstrapping {NUM_OF_BOOTSTRAP_ITER} repeats')
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_to = os.path.join(FIG_DIR, f"{CONFIDENCE}_{full_type_name}_ranks.png")
    plt.savefig(save_to)
    plt.show()

    # # graph 4 - ranks:
    # melted_ranks = pd.melt(ranks)
    # melted_ranks["weights"] = 0
    # for mistake in NUCLE_MISTAKE_TYPES:
    #     val = mistakes["weights"][mistake]
    #     melted_ranks["weights"][melted_ranks["variable"] == mistake] = val
    # melted_ranks.sort_values(by=['mean_rank'], inplace=True)
    # # print("melted ranks: \n",melted_ranks.head(50))
    # bx = sns.barplot(x="variable", y="value", data=melted_ranks)
    # bx.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # bx.set(xlabel=f'{type_name} error categories', ylabel="linear regression weight's rank",
    #        title=f'{type_name} NUCLE error categories ranked weights (with CI) based on bootstrapping 10K repeats')
    # plt.show()


def _coarse_col(col):
    if ":" in col:
        return ":".join(col.split(":")[1:])
    return col


def coarse_errant(sentences):
    """
    converts a sentences df with errant types to coarser types (without U R M)
    :param sentences:
    :return:
    """
    df = pd.DataFrame()
    for col in sentences.columns:
        new_col = _coarse_col(col)
        if new_col in df.columns:
            df[new_col] += sentences[col]
        else:
            df[new_col] = sentences[col]
    assert ['Nucle_ID'] + COARSE_ERRANT_TYPES + ['TotalMistakes', 'z-score'] == list(df.columns)
    return df


def get_db_path(tpe):
    if tpe == AnnotationType.ERRANT or tpe == AnnotationType.COLLAPSED_ERRANT:
        return ERRANT_DB_ADDR
    elif tpe == AnnotationType.SERCL:
        return SERCL_DB_ADDR
    elif tpe == AnnotationType.NUCLE:
        return NUCLE_DB_ADDR
    else:
        raise ValueError(f"Unknown type: {tpe}")


def analyze(df, tpe, force=True, force_bootstrap=True):
    """
    gets df of senteces as vectors and analize them - save stats and plot graphs
    :param df:  pandas df of sentences as a vector of mistakes and a z-score. 0 if using file.
    :param tpe: type used
    :return: mistakes stats
    """
    if df is not None:
        sentences = get_senteces_by_df(get_db_path(tpe), df, tpe,
                                       force=force)  # if changes filter use this instead of loading from file
    else:
        sentences = pd.read_csv(
            get_sentences_mistakes_scores_path(tpe))
    sentences = sentences.loc[~(sentences["TotalMistakes"] == 0), :]
    if tpe == AnnotationType.COLLAPSED_ERRANT:
        sentences = coarse_errant(sentences)
    mistakes = mistakes_stats(sentences)
    mistakes["weights"] = list(get_weights(sentences, False)) + [0]
    mistakes["regularized_weights"] = get_weights(sentences, True)
    all_mistakes, ranks = bootstrap_weights(sentences, tpe, force=force_bootstrap)
    mistakes["upper"], mistakes["lower"], mistakes["upper_rank"], mistakes["lower_rank"], = \
        all_mistakes.transpose()["upper"], all_mistakes.transpose()["lower"], all_mistakes.transpose()[
            "upper_ranks"], all_mistakes.transpose()["lower_ranks"]
    mistakes["mean_rank"] = all_mistakes.transpose()["mean_rank"]
    type_name, full_name = get_saving_name(tpe)
    mistakes.to_csv(f"mistakes_weights_with_regression_{full_name}.csv", sep=",", encoding='utf-8', index=False)
    all_mistakes.drop(["upper", "lower", "lower_ranks", "upper_ranks", "mean_rank"], inplace=True)
    plot_mistakes(all_mistakes, mistakes, ranks, tpe)
    return mistakes


def demo_analyze():
    """
    anlyze-like function, loads the bootstrapped data from file instead of re-creating it.
    :return:
    """
    all_mistakes = pd.read_csv(
        r"NUCLE\my_NUCLE_parser\debug.csv",
        index_col=0)
    mistakes = pd.read_csv(
        r"NUCLE\my_NUCLE_parser\mistakes_weights_with_regression.csv",
        index_col=0)
    ranks = pd.read_csv(
        r"NUCLE\my_NUCLE_parser\ranks.csv",
        index_col=0)
    plot_mistakes(all_mistakes, mistakes, ranks, False)
