import json
import os
import re


class HelperUtils:

    @staticmethod
    def read_jsonl(path: str):
        """Reads a JSONL File and returns a list of Python Objects.

        Args:
            path (str): Path for JSONL File.

        Returns:
            list[any]: List of Objects in the file.
        """
        with open(path) as fh:
            return [json.loads(line) for line in fh.readlines() if line]
