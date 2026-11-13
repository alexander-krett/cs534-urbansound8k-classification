"""Compatibility entry point for the MFCC/SVM baseline."""

import sys

from train_baselines import main


if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "svm"]
    main()
