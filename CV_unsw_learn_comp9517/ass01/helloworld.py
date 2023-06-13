#!/usr/bin/env python


import argparse



parser = argparse.ArgumentParser()

parser.add_argument('num', help='help text', type=str)
# /Users/guohaojin/Downloads/COMP9517/ass01/ass01_output
args = parser.parse_args()
result = args.num

print(result)

