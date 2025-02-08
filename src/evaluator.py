import re
import os


class GSM8K_Evaluator:
    """
    GSM8K_Evaluator class. Handles logic to extract relevant answers and check correctness on GSM8K Benchmark.
    """

    ANSWER_FORMAT: str
    INVALID_ANS: str
    TEST_PATH: str
    TRAIN_PATH: str

    def __init__(self):
        """
        Initializes the Evaluation Variables.
        """
        self.ANSWER_FORMAT = r"#### (\-?[0-9\.\,]+)"
        self.INVALID_ANS = "[invalid]"
        self.TEST_PATH = os.path.join("../gsm8k_data/", "test.jsonl")
        self.TRAIN_PATH = os.path.join("../gsm8k_data/", "train.jsonl")

    def extract_answer(self, model_output: str) -> str:
        """Extract the answer for the given model output. Will be used for comparing solutions.

        Args:
            model_output (str): Model Output for GSM8K Problem.

        Returns:
            str: Extracted Answer (If Any else returns Invalid).
        """
        ANS_RE = re.compile(self.ANSWER_FORMAT)

        match = ANS_RE.search(model_output)
        if match:
            ans_str = match.group(1).strip().replace(",", "")
            return ans_str
        else:
            return self.INVALID_ANS

    def evaluate_gsm8K(self, Model_Output: str, Actual_Output: str) -> bool:
        """Evaluates a specific Test case by comparing actual output with model output.

        Args:
            Model_Output (str): Large Language Model Output for GSM8K Problem.
            Actual_Output (str): Actual Output for the same GSM8K Problem.

        Returns:
            bool: Whether Model correctly predicted the output or not.
        """
        Model_ANS = self.extract_answer(Model_Output)
        Actual_ANS = self.extract_answer(Actual_Output)
        return Model_ANS == Actual_ANS
