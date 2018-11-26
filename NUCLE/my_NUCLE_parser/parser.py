# a NUCLE parser for ME (mistake evaluation) project.

import random;
import regex

ATRIBUTES = "A"
SENTENCE = "S"

# sentence is a tuple as follows: (the nucle original sentence, list of mistakes)

LINE_INX = 0
SENTANCE_INX = 1
MISTAKES_INX = 2

# the list of mistakes and cerrections is as follows: [mistake starting index, mistake end index, mistake type, correction]
NUM_OF_MISTAKE_ATTRIBUTES = 4
ATTRIBUTES_INX = 1


def get_random_inx(lines):
    while True:
        i = random.randint(0, len(lines))
        line = lines[i].split()
        if len(line) > 4 and line[0] == SENTENCE and (
                "http" not in line):  # it's a valid sentence - more then 4 words, and doesn't include http address
            break
    return i


def parse_mistake(line):
    if line.startswith(ATRIBUTES):
        mistake = line.replace('|', ' ').split()
        mistake = mistake[ATTRIBUTES_INX:ATTRIBUTES_INX + NUM_OF_MISTAKE_ATTRIBUTES]  # 1:5 in NUCLE as it is
        mistake[0] = int(mistake[0])
        mistake[1] = int(mistake[1])
        if mistake[3] == 'REQUIRED':
            mistake[3] = " "
        return mistake
    return False


def parse_sentence(lines, i):
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


def main():
    file_name = "/cs/usr/ofirshifman/safe/GEC_ME_PROJECT/NUCLE/release3.2/data/conll14st-preprocessed.m2"  # NUCLE db address
    lines = open(file_name).read().splitlines()
    out_file = open("/cs/usr/ofirshifman/safe/GEC_ME_PROJECT/NUCLE/my_NUCLE_parser/out_file.txt", "w")


    for k in range(20):
        parsed_sentence = (parse_sentence(lines, get_random_inx(lines)))
        line = parsed_sentence[SENTANCE_INX]
        line = line.replace(' ,', ',')  # just to make the sentence look a bit nicer
        line = line.replace(' .', '.')
        line = line.replace('( ', '(')
        line = line.replace(' )', ')')
        line = line.replace(' \'', '\'')
        out_file.write("sentence number " + str(k + 1) + " is originaly in line " + str(
            parsed_sentence[LINE_INX]) + " and is:\n" + line)
        print(line)
        out_file.write("\nmistakes for sentence number " + str(k + 1) + " are:\n")
        for mistake in parsed_sentence[MISTAKES_INX]:
            out_file.write(str(mistake)+ "\n")
        out_file.write('\n')

    out_file.close()


if __name__ == '__main__':
    main()
