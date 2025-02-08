import json
import os
import re
import torch


class HelperUtils:

    @staticmethod
    def read_jsonl(path: str):
        """Reads a JSONL File and returns a list of Python Objects.

        Args:
            path (str): Path for JSONL File.

        Returns:
            list[any]: List of Objects in the file.
        """
        with open(path) as file:
            return [json.loads(line) for line in file.readlines() if line]

    @staticmethod
    def showCurrentMemoryStats():
        """Shows Current Memory Statistics : GPU Type, Max Memory and Reserved Memory."""
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max Memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
