import argparse

class MainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--tokenizer_path', type=str)
        self.parser.add_argument('--model_path', type=str)
        self.parser.add_argument('--num_label', type=int)
        self.parser.add_argument('--num_epoch', type=int)
        self.parser.add_argument('--batch_size', type=int)

        self.parser.add_argument('--train_path', type=str)
        self.parser.add_argument('--test_path', type=str)

        return self.parser.parse_args()
    

class ExtractOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def parse_args(self):
        self.parser.add_argument('--tokenizer_path', type=str)
        self.parser.add_argument('--model_path', type=str)
        self.parser.add_argument('--device', type=str)
        self.parser.add_argument('--batch_size', type=int)
        self.parser.add_argument('--data_path', type=str)

        return self.parser.parse_args()