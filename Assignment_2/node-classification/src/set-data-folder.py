import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", dest="path", type=str, default="./",
                    help="Where to setup the data folder.")
args = parser.parse_args()

os.system(f"unzip data.zip -d {args.path}")
