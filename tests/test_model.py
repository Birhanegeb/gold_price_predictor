"""CLI entry point for evaluating the model on its chronological holdout."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.test_model import main


if __name__ == "__main__":
    main()