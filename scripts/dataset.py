import logging
from typing import Dict, List, Any, Iterator
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def device(use_torch_cuda: Any = torch.cuda) -> str:
    """
    A function to determine if we are using a CPU or CUDA.

    Parameters
    ----------
    use_torch_cuda : Any
        Defaults to torch.cuda, but accepts injected classes for testing.

    Returns
    -------
    str
        Either the string "cuda:0" or "cpu"
    """
    if use_torch_cuda.is_available():
        return "cuda:0"
    return "cpu"


class IntentTrainerDataset(Dataset):
    """
    An implementation of torch.utils.data.Dataset necessary to make the trainer work.
    """

    def __init__(
        self,
        encodings: List[Dict[str, List[int]]],
        labels: List[List[int]],
        device_name: str,
    ):
        """
        Parameters
        ----------
        encodings
            A list of encoded sequences of ints, using the appropriate tokenizer, along with their
            attention masks. See BertIntentClassifier.text_token_vectors to see the exact
            output.
        labels
            A list of ints encoding the classes to train on for each entry in encodings.
        device_name
            Either "cpu" or "cuda:0" See device() above.
        """
        self._encodings = encodings
        self._labels = labels
        self._device = device_name

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for idx in range(len(self._encodings)):
            yield self[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with the vectors, values and labels that the Hugging Face trainer expects.

        Parameters
        ----------
        idx
            An integer index to the desired entry.

        Returns
        -------
        Dict[str, Tensor]
            A collection of labeled tensors with the labels Hugging Face expects.
        """
        item = {}
        for key, val in self._encodings[idx].items():
            item[key] = torch.tensor(val).to(self._device)
        item["labels"] = torch.tensor(self._labels[idx]).to(self._device)
        return item

    def __len__(self) -> int:
        return len(self._labels)
