# Standard
from typing import Optional
import json
import os.path
import random
import uuid

# Third Party
from datasets import Dataset, concatenate_datasets, load_dataset

# First Party
from instructlab.sdg.logger_config import setup_logger
from instructlab.sdg.utils import GenerateException

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


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question_hack(synth_example):
    if "question" in synth_example:
        return synth_example["question"]

    if not synth_example.get("output"):
        raise GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[0].strip() + "?" if len(parts) == 2 else ""


# This is also a hack. See the comment above _get_question.
def _get_response_hack(synth_example):
    if "response" in synth_example:
        return synth_example["response"]

    if "output" not in synth_example:
        raise GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _generate_knowledge_qa_dataset(
    generated_dataset: Dataset, keep_context_separate=False
):
    """
    Generate question and answer pairs from the newly generated dataset
    for each taxonomy leaf node. Each row of the generated dataset gets
    converted to have messages, metadata, and id columns.

    If `keep_context_separate` is True, then a context column is also added.
    If `keep_context_separate` is False, the context colum is omitted and
    the context is instead added directly to the user message content.
    """

    def __create_qa_row(rec):
        msg_id = str(uuid.uuid4())
        context = rec["document"]
        instruction = _get_question_hack(rec)
        response = _get_response_hack(rec)
        metadata = {
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": "document_knowledge_qa",
        }
        if "raw_document" in rec and "dataset_type" in rec:
            metadata.update(
                {
                    "raw_document": rec["raw_document"],
                    "dataset_type": rec["dataset_type"],
                }
            )
        metadata = json.dumps(metadata)
        if keep_context_separate:
            messages = [
                {"role": "user", "content": f"{instruction}"},
                {"role": "assistant", "content": response},
            ]
            return {
                "messages": messages,
                "metadata": metadata,
                "id": msg_id,
                "context": context,
            }
        messages = [
            {"role": "user", "content": f"{context}\n\n{instruction}"},
            {"role": "assistant", "content": response},
        ]

        return {"messages": messages, "metadata": metadata, "id": msg_id}

    knowledge_ds = generated_dataset.map(
        __create_qa_row, remove_columns=generated_dataset.column_names
    )
    return knowledge_ds


def _build_raft_dataset(ds: Dataset, p, num_doc_in_context=4):
    """
    Add additional context to each sample in a knowledge_qa_dataset by
    selecting the context from random other samples and adding that
    combined with this sample's original context all into the user content
    section of the sample's messages.

    This expects to be called with a dataset that has a `context` column,
    such as the output from _generate_knowledge_qa_dataset with the param
    `keep_context_separate` equal to True. When this finishes, the `context`
    column is removed from the dataset and all context moved to the user
    messages.
    """
    all_context = ds["context"]
    all_context = [
        " ".join(e.split(" ")[: random.randint(100, 500)]) for e in all_context
    ]
    ds = ds.add_column("row_idx", range(ds.num_rows))

    def __pick_documents(rec, p):
        # Loop until we find enough other documents to add to the context
        # for this document. Exit the loop early if we have fewer total
        # documents than the number of documents we want in our context
        # so that we don't end up looping forever. This handles edge
        # cases where the number of generated instructions is very low,
        # like in CI or user's testing small sizes.
        while True:
            selected_docs = random.choices(range(ds.num_rows), k=num_doc_in_context)
            if ds.num_rows <= num_doc_in_context:
                break
            if rec["row_idx"] not in selected_docs:
                break
        if random.uniform(0, 1) < p:
            docs = [
                all_context[idx] for idx in selected_docs[: num_doc_in_context - 1]
            ] + [rec["context"]]
            # rec['indicator'] ='golden'
        else:
            docs = [all_context[idx] for idx in selected_docs]
            # rec['indicator'] = 'distractor'
        random.shuffle(docs)
        docs = "\n".join(([f"Document:\n{e}\n\n" for idx, e in enumerate(docs)]))
        user_idx, user_msg = [
            (idx, rec_msg)
            for idx, rec_msg in enumerate(rec["messages"])
            if rec_msg["role"] == "user"
        ][0]
        user_inst = user_msg["content"]
        rec["messages"][user_idx]["content"] = f"{docs}\n\n{user_inst}"
        rec["messages"] = rec["messages"]
        metadata = json.loads(rec["metadata"])
        metadata["dataset"] += f"_raft_p{p}"
        rec["metadata"] = json.dumps(metadata)
        return rec

    ds = ds.map(__pick_documents, fn_kwargs={"p": p}, remove_columns=["context"])
    return ds


def _conv_pretrain(rec):
    """
    Convert a messages dataset that contains only user/assistant entries per
    message (and in that order) to a pretraining message used downstream by
    the training pipeline. `_generate_knowledge_qa_dataset` creates the type
    of dataset expected here.
    """
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def _create_phase10_ds(generated_dataset: Dataset):
    """
    Create a dataset for Phase 1.0 of downstream training.

    This dataset is in our messages format, with each sample having
    additional context mixed in from other samples to improve the
    training outcomes.
    """
    knowledge_ds = _generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=True
    )
    knowledge_ds = _build_raft_dataset(knowledge_ds, p=0.4)

    return knowledge_ds


def _create_phase07_ds(generated_dataset: Dataset):
    """
    Create a dataset for Phase 0.7 of downstream training.

    Phase 0.7 is a pretraining phase, and this dataset contains messages
    with a special `pretraining` role used by downstream training before
    running the full training with the Phase 1.0 dataset.
    """
    # Phase 0.7
    knowledge_ds = _generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=False
    )
    knowledge_ds = knowledge_ds.map(_conv_pretrain)

    return knowledge_ds


def _convert_to_leaf_node_messages(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain a 'messages' column required
    for training.

    Note that this is for the new messages format, introduced with data
    mixing.
    """
    user_query = _unescape(_get_question_hack(sample))
    response = _unescape(_get_response_hack(sample))

    sample["id"] = str(uuid.uuid4())
    sample["messages"] = [
        {"content": sys_prompt, "role": "system"},
        {"content": user_query, "role": "user"},
        {"content": response, "role": "assistant"},
    ]

    return sample


class DataMixer:
    # pylint: disable=too-many-instance-attributes

    # This determines how many samples to pick from each skill when mixing
    # skill datasets. It's only used for skills, as knowledge may require
    # a variable number of samples depending on the length of the
    # knowledge documents in question. The expectation is that this is
    # enough samples to sufficiently learn a new skill while also ensuring
    # a balance of overall mixed data when learning multiple skills at
    # once.
    NUM_SYNTH_SKILLS = 30

    def __init__(self, output_dir, date_suffix, sys_prompt, num_procs):
        self.output_dir = output_dir
        self.sys_prompt = sys_prompt
        self.date_suffix = date_suffix
        self.num_procs = num_procs

        self.knowledge_recipe = Recipe(sys_prompt=self.sys_prompt)
        self.skills_recipe = Recipe(sys_prompt=self.sys_prompt)

        self.output_file_mixed_knowledge = f"knowledge_train_msgs_{date_suffix}.jsonl"
        self.output_file_mixed_skills = f"skills_train_msgs_{date_suffix}.jsonl"

    def _gen_leaf_node_data(
        self, leaf_node_data, recipe, output_file_leaf_node, sampling_size=1.0
    ):
        """
        Write the data generated from each taxonomy leaf node to a file.
        Later on, after all data is generated, the data mixing will read data
        from these files to generate the overall mixed dataset.
        """
        output_file = os.path.join(self.output_dir, output_file_leaf_node)
        leaf_node_data.to_json(output_file, orient="records", lines=True)
        recipe.add_dataset(output_file_leaf_node, sampling_size)

    def collect(self, leaf_node_path, new_generated_data, is_knowledge):
        if is_knowledge:
            knowledge_phase_data = _create_phase07_ds(new_generated_data)
            output_file_leaf_knowledge = (
                f"node_datasets_{self.date_suffix}/{leaf_node_path}_p07.jsonl"
            )
            self._gen_leaf_node_data(
                knowledge_phase_data,
                self.knowledge_recipe,
                output_file_leaf_knowledge,
            )

            skills_phase_data = _create_phase10_ds(new_generated_data)
            output_file_leaf_skills = (
                f"node_datasets_{self.date_suffix}/{leaf_node_path}_p10.jsonl"
            )
            self._gen_leaf_node_data(
                skills_phase_data,
                self.skills_recipe,
                output_file_leaf_skills,
            )
        else:
            messages = new_generated_data.map(
                _convert_to_leaf_node_messages,
                fn_kwargs={"sys_prompt": self.sys_prompt},
                num_proc=self.num_procs,
            )
            output_file_leaf = (
                f"node_datasets_{self.date_suffix}/{leaf_node_path}.jsonl"
            )
            self._gen_leaf_node_data(
                messages,
                self.skills_recipe,
                output_file_leaf,
                sampling_size=self.NUM_SYNTH_SKILLS,
            )

    def _gen_mixed_data(self, recipe, output_file_mixed):
        """
        Mix the generated leaf node data into a single dataset and write it to
        disk. The heavy lifting is delegated to the Recipe class.
        """
        if recipe.dataset_added:
            recipe.save_mixed_dataset(
                os.path.join(self.output_dir, output_file_mixed),
                self.num_procs,
            )

    def generate(self):
        self._gen_mixed_data(self.knowledge_recipe, self.output_file_mixed_knowledge)
        self._gen_mixed_data(
            self.skills_recipe,
            self.output_file_mixed_skills,
        )
