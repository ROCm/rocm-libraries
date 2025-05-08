#!/usr/bin/env python3

import os
import subprocess
import sys
import argparse
import tempfile


def main():
    print("main")

    parser = argparse.ArgumentParser(prog='rocfft_mpi_test')

    parser.add_argument('--worker',
                        type=str,
                        default=None,
                        help='mpi worker path')
    parser.add_argument('--rocffttest',
                        type=str,
                        default=None,
                        help='path to rocfft-test')
    parser.add_argument('--launcher', type=str, default=None, help='FIXME')
    parser.add_argument('--gpuidvar', type=str, default=None, help='FIXME')
    parser.add_argument('--gpusperrank',
                        type=int,
                        help='Gpus per rank',
                        default=1)
    parser.add_argument('--timeout',
                        type=int,
                        help='timeout in seconds',
                        default=5 * 60)
    parser.add_argument('--nranks',
                        type=int,
                        help='number of ranks',
                        required=True)

    args = parser.parse_args()

    print("nranks", args.nranks)
    print("gpus per rank", args.gpusperrank)

    print("Running token generation:")

    tokencmd = [
        args.rocffttest, "--gtest_filter=*multi_gpu*-*adhoc*", "--mp_lib",
        "mpi", "--mp_ranks",
        str(args.nranks), "--mp_launch", "foo", "--printtokens", "--ngpus",
        str(1)
    ]

    print(tokencmd)

    fout = tempfile.TemporaryFile(mode="w+")
    ptoken = subprocess.Popen(tokencmd, stdout=fout, stderr=subprocess.STDOUT)
    ptoken.wait()

    fout.seek(0)
    cout = fout.read()

    if ptoken.returncode != 0:
        print("token generation failed")
        print(cout)
        sys.exit(1)

    tokens = []
    foundtokenstart = False
    for line in cout.splitlines():
        if foundtokenstart:
            tokens.append(line)
        if line.startswith("Tokens:"):
            foundtokenstart = True

    print("Generated", len(tokens), "tokens")

    failedtokens = []
    for idx, token in enumerate(tokens):
        print("testing token", idx)
        workercmd = [args.worker] + ["--token", token]
        if args.gpuidvar != None:
            bashprecmd = "export ROCR_VISIBLE_DEVICES="
            for idx in range(args.gpusperrank):
                if (idx != 0):
                    bashprecmd += ","
                bashprecmd += "$(( " + str(
                    args.gpusperrank) + " * " + args.gpuidvar + " + " + str(
                        idx) + " ))"
            bashprecmd += "; "
            cmd = ["bash", "-c", bashprecmd + " " + " ".join(workercmd)]
        else:
            cmd = workercmd

        allcmd = [args.launcher] + cmd
        print(allcmd)

        fout = tempfile.TemporaryFile(mode="w+")
        p = subprocess.Popen(allcmd, stdout=fout, stderr=subprocess.STDOUT)
        p.wait(args.timeout)

        fout.seek(0)
        cout = fout.read()
        print(cout)

        if p.returncode == 0:
            print("PASSED")
        else:
            print("FAILED")
            failedtokens.append(token)

    print("Failed", len(failedtokens), "tokens:")
    for token in failedtokens:
        print(token)

    print("Ran", len(tokens), "tokens, with", len(failedtokens), "failures.")

    sys.exit(len(failedtokens))


if __name__ == '__main__':
    main()
