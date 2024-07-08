# Standard
from typing import Optional
import json

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset

# First Party
from instructlab.sdg.logger_config import setup_logger

ALLOWED_COLS = ["id", "messages", "metadata"]
logger = setup_logger(__name__)


def _adjust_train_sample_size(ds: Dataset, num_samples: int):
    """
    Return a dataset with num_samples random samples selected from the
    original dataset.
    """
    logger.info(f"Rebalancing dataset to have {num_samples} samples ...")
    df = ds.to_pandas()
    df = df.sample(n=num_samples, random_state=42, replace=True).reset_index(drop=True)
    return Dataset.from_pandas(df)


def _load_ds(path, sampling_size, num_proc):
    """
    Load a dataset from the given file path and select sampling_size
    number/ratio of samples from it, ensuring the loaded dataset has only
    ALLOWED_COLS columns in it with any additional columns moved to the
    metadata section.
    """
    logger.info(f"Loading dataset from {path} ...")
    dataset = load_dataset("json", data_files=path, split="train")
    logger.info(f"Dataset columns: {dataset.column_names}")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    if sampling_size != 1.0:
        if isinstance(sampling_size, int):
            num_samples = sampling_size
        else:
            num_samples = int(len(dataset) * sampling_size)
        dataset = _adjust_train_sample_size(dataset, num_samples)

    # move any column that is not in ALLOWED_COLS to metadata
    def _move_unallowed_cols_to_metadata(example):
        metadata = example.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        for col in dataset.column_names:
            if col not in ALLOWED_COLS:
                metadata[col] = example[col]
                example.pop(col)
        example["metadata"] = json.dumps(metadata)
        return example

    dataset = dataset.map(_move_unallowed_cols_to_metadata, num_proc=num_proc)

    # check if metadata column is string if not convert it using json.dumps
    if not isinstance(dataset["metadata"][0], str):
        dataset = dataset.map(
            lambda x: {"metadata": json.dumps(x["metadata"])}, num_proc=num_proc
        )

    return dataset


def _add_system_message(sample: dict, sys_prompt: str) -> dict:
    """
    Ensure every sample has a system message with the correct system prompt
    """
    # check if the messages have role system
    has_system = False
    for msg in sample["messages"]:
        if msg["role"] == "system":
            has_system = True
            msg["content"] = sys_prompt

    if not has_system:
        sample["messages"].insert(0, {"role": "system", "content": sys_prompt})

    return sample


class Recipe:
    """
    A Recipe describes how datasets were mixed, including the path and
    sampling size used for each included dataset as well as the system
    prompt used when generating the data in those datasets.
    """

    def __init__(
        self, initial_datasets: Optional[list] = None, sys_prompt: Optional[str] = ""
    ):
        self.recipe = {
            "datasets": initial_datasets or [],
            "sys_prompt": sys_prompt,
        }
        self.sys_prompt = self.recipe.get("sys_prompt", "")
        self.dataset_added = False

    def _create_mixed_dataset(self, num_proc):
        """
        Create the mixed dataset from its list of included datasets, taking
        into account the desired sampling size from each individual dataset
        to control the overall mix of samples in the final dataset.
        """
        if not self.dataset_added:
            logger.error("No dataset added to the recipe")

        mixed_ds = [
            _load_ds(dataset["path"], dataset["sampling_size"], num_proc)
            for dataset in self.recipe["datasets"]
        ]

        mixed_ds = concatenate_datasets(mixed_ds)
        mixed_ds = mixed_ds.map(
            _add_system_message,
            fn_kwargs={"sys_prompt": self.sys_prompt},
            num_proc=num_proc,
        )

        # assert that the dataset only has the allowed columns
        assert set(mixed_ds.column_names) == set(
            ALLOWED_COLS
        ), "Dataset has invalid columns"
        return mixed_ds

    def add_dataset(self, path, sampling_size):
        """
        Add a dataset to this recipe.

        Args:
            path: The file path to the dataset's samples, as jsonl
            sampling_size: An int or float that specifices the number of
                           samples (if int) or the ratio of samples (if
                           float) to include in the mixed dataset. A value
                           of 1.0 means include all samples, 0.5 means half
                           of the samples, and so on.
        """
        self.dataset_added = True
        self.recipe["datasets"].append({"path": path, "sampling_size": sampling_size})

    def save_mixed_dataset(self, output_path, num_proc):
        """
        Create the mixed dataset and write it to the specified output path
        as a jsonl file.
        """
        mixed_ds = self._create_mixed_dataset(num_proc)
        mixed_ds.to_json(output_path, orient="records", lines=True)
        logger.info(f"Mixed Dataset saved to {output_path}")
