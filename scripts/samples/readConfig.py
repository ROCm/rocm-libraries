import sys, json

import argparse

def replace_string(line, str1, str2):
    index0 = line.find(str1)
    index1 = line[index0+len(str1):].find(' ')

    if index0 == -1:
        return line

    if index1 == -1:
        line = line[:index0] + str2
    else:
        index1 += index0 + len(str1)
        line = line[:index0] + str2 + ' ' + line[index1:]
    return line

def modify(line):
    # remove --bench-x
    line = line.replace('--bench-x ', '')

    # remove --bench-n X
    line = replace_string(line, '--bench-n ', '')

    # replace --rocalution X to --rocalution filename_string
    line = replace_string(line, '--rocalution ', '--rocalution $filename')

    # replace --blockdim X to --blockdim $blockdim
    line = replace_string(line, '--blockdim ', '--blockdim $blockdim')

    # replace --sizen X to --sizen $sizen
    line = replace_string(line, '--sizen ', '--sizen $sizen')

    return line

def main():
    parser = argparse.ArgumentParser("readConfig")
    parser.add_argument('filename')
    args = parser.parse_args()

    print(modify(json.load(open(args.filename))['cmdlines'][0]))

main()
