# a NUCLE parser for ME (mistake evaluation) project.

import sys;
import random;

ATRIBUTES = "A"
SENTENCE = "S"







def get_sentance(lines):
    line = "X" #some fale value for the do-while
    while line.startswith(SENTENCE):
        i = random.randint(0,lines.len)
        line = lines[i]
    return line

def parse_sentance():


def main():
    file_name = sys.argv[1]  # NUCLE db address
    lines = open(file_name).read().splitlines();
    for i



if __name__ == '__main__':
    main();
