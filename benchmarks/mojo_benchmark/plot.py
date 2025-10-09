"""
Take a Tritonbench output csv and plot the results.
"""

import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="CSV file to plot")
    parser.add_argument("--output", help="Output file name")
