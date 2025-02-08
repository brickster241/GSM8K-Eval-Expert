import re
import torch
from transformers import (
    StoppingCriteria,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


class CalculatorStoppingCriteria(StoppingCriteria):
    """
    CalculatorStoppingCriteria Class.  Will detect placeholders (<<expr>>), compute results,
    and modifies the output before continuing generation.
    """

    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    intermediate_results: dict
    EXPR_FORMAT: str

    def __init__(self, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast):
        """
        Initializes the Tokenizer that will be used for StoppingCriteria class Implementation.

        Args:
            tokenizer (AutoTokenizer): Tokenizer for the Large Language Model.
        """
        self.EXPR_FORMAT = r"<<(.*?)>>"
        self.tokenizer = tokenizer
        self.intermediate_results = {}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        Modifies the Generation Process. Checks if model is trying to output <<expr>>.
        If yes, computes the result of expr and modifies output before continuing generation.

        Args:
            input_ids (torch.LongTensor): _description_
            scores (torch.FloatTensor): _description_
        """

        decoded_text = self.tokenizer.decode(input_ids[0])  # Get Generated Text so far.
        regex_match = re.search(self.EXPR_FORMAT, decoded_text)
        if regex_match:
            expr = regex_match.group(1)
            if expr not in self.intermediate_results:
                expr_result = self.Calculate(expr)
                self.intermediate_results[expr] = expr_result

            # Replace expression with computed value.
            append_text = f"{self.intermediate_results[expr]}"

            new_tokens = self.tokenizer.encode(append_text, return_tensors="pt")

            # Replace Old Input IDs with new ones
            return torch.cat([input_ids, new_tokens], dim=-1)

        # If No Expression present, continue normally
        return False

    def Calculate(expression: str) -> str:
        """
        Safely Evaluates an Arithmetic Expression and returns output in a string format.
        Args:
            expression (str): The Arithmetic Expression which needs to be calculated.

        Returns:
            str: Evaluated Result or Error String.
        """
        try:
            return str(eval(expression))
        except Exception as exc:
            return f"ERROR : {exc}"
