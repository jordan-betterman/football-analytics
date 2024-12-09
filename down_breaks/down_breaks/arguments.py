import argparse
import os


class ArgumentsBase(object):
    def __init__(self):
        self.name = argparse.Namespace()
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialized = False
        self.current_path = os.path.join(os.getcwd(), "perspective_transform")
        self.base_path = os.getcwd()

    def main_args_initialization(self):
        self.parser.add_argument(
            "--predictor",
            type=str,
            default="Runpass",
            help="variable you want to predict",
        )
        self.parser.add_argument("--file", type=str)
        self.parser.add_argument("--team", type=str, help="team name")
        self.parser.add_argument(
            "--output",
            type=str,
            default="/Users/jordanbetterman/Desktop/down-breaks",
            help="output folder",
        )
        self.parser.add_argument(
            "--year", type=str, default="2023", help="year of dataset"
        )


class Arguments(ArgumentsBase):
    def __init__(self):
        super().__init__()
        self.main_args_initialization()

    def parse(self):
        opt = self.parser.parse_args()

        return opt
