# a NUCLE parser for ME (mistake evaluation) project.
import csv
import random
import pandas as pd

# from tqdm import tqdm

# sentence is a tuple as follows: (the nucle original sentence, list of mistakes)
# -------- Constants: --------
MIN_WORDS = 7  # minimum words per sentence
NUM_OF_BATCHES = 290  # acprocimetly taking all of NUCLE (after some cleaning)
SENTENCES_PER_BATCH = 100
CTROL_SENTENCES = 200
PERFECT_RATE = 0.15
CTROL_RATE = 0.15  # per single batch
CTROL_PERFECT_RATE = 0  # 0 means that for now there are no perfect control sentences
M_RATE = 1 - PERFECT_RATE - CTROL_RATE
P_PER_BATCH = int(PERFECT_RATE * SENTENCES_PER_BATCH)
M_PER_BATCH = int(M_RATE * SENTENCES_PER_BATCH)
C_PER_BATCH = int(CTROL_RATE * SENTENCES_PER_BATCH)
ATRIBUTES = "A"
SENTENCE = "S"
LINE_INX = 0
SENTANCE_INX = 1
MISTAKES_INX = 2
ATTRIBUTES_INX = 1
NUM_OF_MISTAKE_ATTRIBUTES = 4

# files addresses:

BASE_ADDR = "NUCLE/"
MY_P_BASE_ADDR = BASE_ADDR + "my_NUCLE_parser/csvs/"
TO_MTURK_BASE_ADDR = MY_P_BASE_ADDR
M_FILE_ADDR = MY_P_BASE_ADDR + "m_sentences"
C_FILE_ADDR = MY_P_BASE_ADDR + "c_sentences"
P_FILE_ADDR = MY_P_BASE_ADDR + "p_sentences"
FINAL_M_FILE_ADDR = TO_MTURK_BASE_ADDR + "m_sentences.csv"
FINAL_C_FILE_ADDR = TO_MTURK_BASE_ADDR + "c_sentences.csv"
FINAL_P_FILE_ADDR = TO_MTURK_BASE_ADDR + "p_sentences.csv"
M_CSV_ADDR = M_FILE_ADDR + ".csv"
C_CSV_ADDR = C_FILE_ADDR + ".csv"
P_CSV_ADDR = P_FILE_ADDR + ".csv"
MTURK_CSV_ADDR = TO_MTURK_BASE_ADDR + "mTurk_csv.csv"
NUCLE_DB_ADDR = r"NUCLE/release3.2/data/conll14st-preprocessed.m2"  # NUCLE db address

# if these are found in the sentence, we won't use it:
BAD_CHARS = {";", "*", "[", "]", "&"}
HEADERS = ["batch index", "sentence NUCLE ID", "original_text", "is perfect",
           "original_language"]

# -------- Global Vars: --------
not_chosen_m_sentensces = set()
not_chosen_p_sentensces = set()


# the list of mistakes and cerrections is as follows: [mistake starting index, mistake end index, mistake type, correction]

def count_sentences(file_name):
    """
    count how many sentences are in the corpus, and how many attributes per each
    :param file_name:
    """
    lines = open(file_name).read().splitlines()
    flag = False
    atr = 0
    for i in range(len(lines)):
        if flag == True:
            if lines[i].startswith(ATRIBUTES):
                not_chosen_m_sentensces.add(i - 1)
            else:
                not_chosen_p_sentensces.add(i - 1)
        if lines[i].startswith(SENTENCE):
            flag = True
        else:
            flag = False
        if lines[i].startswith(ATRIBUTES):
            atr += 1



def create_data_sets(file_name):
    """ choose sentenses randomly and create 3 Data sets: m_sentences, c_sentences, p_sentences: """
    # open files:
    lines = open(file_name).read().splitlines()
    txt_sentences_file = open(M_FILE_ADDR + ".txt", "w")
    txt_perfect_file = open(P_FILE_ADDR + ".txt", "w")
    csv_m_file = open(M_CSV_ADDR, "w")
    csv_p_file = open(P_CSV_ADDR, "w")
    csv_c_file = open(C_CSV_ADDR, "w")

    # create writers:
    csv_sentence_writer = csv.writer(csv_m_file)
    csv_perfect_writer = csv.writer(csv_p_file)
    ctrl_txt_sentences_file = open(C_FILE_ADDR + ".txt", "w")
    ctrl_txt_perfect_file = open(MY_P_BASE_ADDR + "ctrl_p_sentences.txt", "w")
    ctrl_csv_sentence_writer = csv.writer(csv_c_file)
    ctrl_csv_perfect_writer = csv.writer(open(MY_P_BASE_ADDR + "ctrl_p_sentences.csv", "w"))

    # write headers:
    csv_sentence_writer.writerow(HEADERS)
    csv_perfect_writer.writerow(HEADERS)
    ctrl_csv_sentence_writer.writerow(HEADERS)
    ctrl_csv_perfect_writer.writerow(HEADERS)

    # write files:
    write_section_out(lines, csv_sentence_writer, txt_sentences_file,
                      SENTENCES_PER_BATCH * (M_RATE), NUM_OF_BATCHES, True)
    write_section_out(lines, csv_perfect_writer, txt_perfect_file,
                      SENTENCES_PER_BATCH * PERFECT_RATE, NUM_OF_BATCHES, False)
    write_section_out(lines, ctrl_csv_sentence_writer, ctrl_txt_sentences_file,
                      (1 - CTROL_PERFECT_RATE) * CTROL_SENTENCES, 1, True)
    write_section_out(lines, ctrl_csv_perfect_writer, ctrl_txt_perfect_file,
                      CTROL_SENTENCES * (CTROL_PERFECT_RATE), 1, False)

    # close files:
    txt_perfect_file.close()
    txt_sentences_file.close()
    ctrl_txt_perfect_file.close()
    ctrl_txt_sentences_file.close()
    csv_m_file.close()
    csv_p_file.close()
    csv_c_file.close()


def get_random_inx(lines, has_mistakes):
    """
    return an available index for a perfect sentence if has_mistakes == False,
    and for a sentence with mistakes else
    :param lines: all lines as an array
    :param has_mistakes: is this index for perfect sentence (=False) ar with mistakes (=True).
    :return: the chosen line index
    """
    while True:
        legal = True
        if has_mistakes:
            i = random.sample(not_chosen_m_sentensces, 1)[0]
        else:
            i = random.sample(not_chosen_p_sentensces, 1)[0]
        line = lines[i].split()
        for c in BAD_CHARS:
            if c in line:
                legal = False
                break
        for word in line:
            if "http" in word:
                legal = False
                break
        if has_mistakes:  # remove the senteeces from available set:
            not_chosen_m_sentensces.remove(i)
        else:
            not_chosen_p_sentensces.remove(i)
        if (not line[1].startswith("(")) and legal and ((len(
                line) > MIN_WORDS) or has_mistakes):  # it's a valid sentence - more then MIN_WORDS words, and doesn't include http address, doesnt have illigal chars and hasn't been chosen yet
            break
    return i


def parse_mistake(line):
    """
    parse a mistake line
    :param line: a mistake attribute line
    :return: the parsed line
    """
    if line.startswith(ATRIBUTES):
        mistake = line.replace('|', ' ').split()
        mistake = mistake[
                  ATTRIBUTES_INX:ATTRIBUTES_INX + NUM_OF_MISTAKE_ATTRIBUTES]  # 1:5 in NUCLE as it is
        mistake[0] = int(mistake[0])
        mistake[1] = int(mistake[1])
        if mistake[3] == 'REQUIRED':
            mistake[3] = " "
        return mistake
    return False


def parse_sentence(lines, i):
    """
    parse a single sentence - original text and mistakes
    :param lines: all lines as an array
    :param i: line number to be parsed
    :return: the parsed line
    """
    line = lines[i][2:]
    sentence = (i + 1, line, [])
    while True:
        i += 1
        cur_line = lines[i]
        mistake = parse_mistake(cur_line)
        if mistake:  # True if parsed a mistake
            sentence[MISTAKES_INX].append(mistake)
        else:  # last sentence to be parsed was not a mistake - end of mistakes for this sentence
            break
    return sentence


def write_section_out(lines, csv_writer, out_text_file, sentences_per_batch,
                      num_of_batches, has_mistakes):
    """
    writes to a csv and txt files a hole section (e.g. perfect or control sentences)
    :param lines: all lines as an array
    :param csv_writer: the writer of the relevant csv file
    :param out_text_file: the writer of the relevant txt file
    :param sentences_per_batch: number of sentences per batch
    :param num_of_batches: number of batches to be writen
    :param has_mistakes: are these perfect sentences (=False) ar with mistakes (=True).
    """
    for i in range(num_of_batches):
        k = 0
        while (k < sentences_per_batch):
            print(i, k)
            parsed_sentence = (parse_sentence(lines, get_random_inx(lines, has_mistakes)))
            is_perfect = (len(parsed_sentence[MISTAKES_INX]) == 0)
            line = parsed_sentence[SENTANCE_INX]
            line = line.replace(' ,', ',')  # just to make the sentence look a bit nicer
            line = line.replace(' .', '.')
            line = line.replace('( ', '(')
            line = line.replace(' )', ')')
            line = line.replace(' \'', '\'')
            line = line.replace(' !', '!')
            line = line.replace(' %', '%')
            line = line.replace('$ ', '$')
            line_str_inx = str(parsed_sentence[LINE_INX])
            if (is_perfect == has_mistakes) or (len(line.split()) < MIN_WORDS):  # if looking
                #  for sentances with mistakes, don't write perfect ones and the same other way around. + sentence must have MIN_WORDS
                continue
            k += 1
            out_text_file.write(
                "sentence number " + str(k) + " in batch number: " + str(
                    i + 1) + " is originaly in line " + line_str_inx + " and is:\n" + line)
            csv_writer.writerow(
                [str(i + 1), line_str_inx, line, not has_mistakes, "english"])
            out_text_file.write(
                "\nmistakes for sentence number " + str(k + 1) + " are:\n")
            for mistake in parsed_sentence[MISTAKES_INX]:
                out_text_file.write(str(mistake) + "\n")
            out_text_file.write('\n')
    return


# ------- create the M-Turk format csv out of the 3 different data sets: m_sentences, c_sentences, p_sentences: -------

def create_mTurk_csv(p_sentences, m_sentences, c_sentences):
    """
    creats the main csv to be posted on mturk.
    :param p_sentences: list of perfect sentences
    :param m_sentences: list of sentences with mistakes
    :param c_sentences: list of control sentences.
    :return:
    """
    # open file and write the headers:
    mTurk_csv_file = open(MTURK_CSV_ADDR, "w")
    mTurk_csv_writer = csv.writer(mTurk_csv_file)
    headers = ["batch"] + [(",Nucle_ID" + str(i) + ",original_text" + str(
        i) + ",is_perfect" + str(i) +
                            ",is_control" + str(i)) for i in
                           range(1, SENTENCES_PER_BATCH + 1)]
    headers = "".join(headers).split(",")
    mTurk_csv_writer.writerow(headers)

    # write csv - every HIT is a line
    for i in range(NUM_OF_BATCHES):
        c_batch = get_c_batch(c_sentences)
        write_batch(i, p_sentences[i * P_PER_BATCH: (i + 1) * P_PER_BATCH],
                    m_sentences[(i * M_PER_BATCH): (i + 1) * M_PER_BATCH],
                    c_batch, mTurk_csv_writer)
    mTurk_csv_file.close()


def get_c_batch(c_sentences):
    """
    create a single batch of randomly chosen control sentences
    :param c_sentences: a list of the control sentences
    :return: a list of C_PER_BATCH (=15) control sentences
    """
    c_batch = []
    c_batch_indexes = set()
    while (len(c_batch) < C_PER_BATCH):
        k = random.randint(0, CTROL_SENTENCES - 1)
        if k not in c_batch_indexes:
            # print(k, len(c_sentences))
            c_batch.append(c_sentences[k])
            c_batch_indexes.add(k)
    return c_batch


#
def get_special_inx():
    """
    get random indexes for both control sentences and perfect sentences
    :return: sentences indexes
    """
    p_indexes = set()
    c_indexes = set()
    while len(p_indexes) < (PERFECT_RATE * SENTENCES_PER_BATCH):
        p_indexes.add(random.randint(0, SENTENCES_PER_BATCH - 1))
    while len(c_indexes) < (CTROL_RATE * SENTENCES_PER_BATCH):
        inx = random.randint(0, SENTENCES_PER_BATCH - 1)
        if inx not in p_indexes:
            c_indexes.add(inx)
    return p_indexes, c_indexes



def write_batch(batch_num, p_sentences, m_sentences, c_sentences, csv_writer):
    """
    write a single batch to the mturk csv finel file
    :param batch_num: batch number
    :param p_sentences: a list of the perfect sentences
    :param m_sentences: a list of the sentences with mistakes
    :param c_sentences: a list of the control sentences
    :param csv_writer: the output csv file writer
    :return:
    """
    p_indexes, c_indexes = get_special_inx()
    cur_p, cur_m, cur_c = 0, 0, 0
    line = [batch_num]
    for i in range(SENTENCES_PER_BATCH):
        if i in p_indexes:
            line = line + (p_sentences[cur_p][1:4]) + ["False"]
            cur_p += 1
        elif i in c_indexes:
            line = line + (c_sentences[cur_c][1:4]) + ["True"]
            cur_c += 1
        else:
            line = line + (m_sentences[cur_m][1:4]) + ["False"]
            cur_m += 1
    csv_writer.writerow(line)
    text = clean_file(open(MTURK_CSV_ADDR).read().splitlines())
    with open(MTURK_CSV_ADDR, "w") as output:
        for line in text:
            output.write(line + '\n')


def csv_to_arr(addr):
    """
    reads a csv to an array
    :param addr: csv file address
    :return:
    """
    with open(addr) as csv_file:
        reader = csv.reader(csv_file)
        return [row for row in reader]


def clean_file(sentences):
    """
    clean file from empty lines
    :param sentences:
    :return:
    """
    return [x for x in sentences if x]


def write_to_mturk():
    """create MTurk csv"""
    p_sentences = clean_file(csv_to_arr(FINAL_P_FILE_ADDR)[1:])  # cut out the headers
    m_sentences = clean_file(csv_to_arr(FINAL_M_FILE_ADDR)[1:])
    c_sentences = clean_file(csv_to_arr(FINAL_C_FILE_ADDR)[1:])
    create_mTurk_csv(p_sentences, m_sentences, c_sentences)


def create_sections():
    """create data sets (put in comment if want to keep the files as the are):"""
    count_sentences(NUCLE_DB_ADDR)
    create_data_sets(NUCLE_DB_ADDR)





if __name__ == '__main__':
    pass


