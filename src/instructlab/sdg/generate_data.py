# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Optional
import json
import os
import random
import time
import uuid

# Third Party
# instructlab - All of these need to go away (other than sdg) - issue #6
from datasets import Dataset
import httpx
import openai
import platformdirs

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg.datamixing import Recipe
from instructlab.sdg.eval_data import generate_eval_task_data, mmlubench_pipe_init
from instructlab.sdg.llmblock import MODEL_FAMILY_MERLINITE, MODEL_FAMILY_MIXTRAL
from instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    SIMPLE_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)
from instructlab.sdg.sdg import SDG
from instructlab.sdg.utils import GenerateException, models
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)

_SYS_PROMPT = "I am, Red Hat® Instruct Model based on Granite 7B, an AI language model developed by Red Hat and IBM Research, based on the Granite-7b-base language model. My primary function is to be a chat assistant."

# This determines how many samples to pick from each skill when mixing
# skill datasets. It's only used for skills, as knowledge may require
# a variable number of samples depending on the length of the
# knowledge documents in question. The expectation is that this is
# enough samples to sufficiently learn a new skill while also ensuring
# a balance of overall mixed data when learning multiple skills at
# once.
NUM_SYNTH_SKILLS = 30


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question(logger, synth_example):
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
def _get_response(logger, synth_example):
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


def _convert_to_messages(sample):
    """
    Convert a sample dictionary to contain 'messages' and 'metadata' columns required for training.

    Note that this is for the legacy messages format, used before data
    mixing was introduced. Once we can drop the older `messages_*.jsonl`
    output files, this can go away.
    """
    # Create user query message
    user_query = sample["inputs"]
    # TODO: in the future we can remove the combinecolumnsblock and combine them here for simplicity
    # if "context" in sample:
    #     user_query = f"{sample['context']}\n\n{sample['inputs']}"

    sample["messages"] = [
        {"content": user_query, "role": "user"},
        {"content": sample["targets"], "role": "assistant"},
    ]
    metadata = {
        key: value
        for key, value in sample.items()
        if key not in ["messages", "inputs", "targets"]
    }
    sample["metadata"] = json.dumps(metadata)

    # keeping required keys for messages training format
    sample = {"messages": sample["messages"], "metadata": sample["metadata"]}

    return sample


def _convert_to_leaf_node_messages(sample: dict, logger, sys_prompt: str):
    """
    Convert a sample dictionary to contain a 'messages' column required
    for training.

    Note that this is for the new messages format, introduced with data
    mixing.
    """
    user_query = _unescape(_get_question(logger, sample))
    response = _unescape(_get_response(logger, sample))

    sample["id"] = str(uuid.uuid4())
    sample["messages"] = [
        {"content": sys_prompt, "role": "system"},
        {"content": user_query, "role": "user"},
        {"content": response, "role": "assistant"},
    ]

    return sample


def _gen_train_data(
    logger, machine_instruction_data, output_file_train, output_file_messages
):
    """
    Generate training data in the legacy system/user/assistant format
    used in train_*.jsonl as well as the legacy messages format used
    in messages_*.jsonl files.

    This can be dropped once we no longer need those formats and are fully
    using the new data mixing messages format.
    """
    train_data = []
    messages_data = []

    for output_dataset in machine_instruction_data:
        for synth_example in output_dataset:
            logger.debug(synth_example)
            user = _get_question(logger, synth_example)
            if len(synth_example.get("context", "")) > 0:
                user += "\n" + synth_example["context"]
            assistant = _unescape(_get_response(logger, synth_example))
            train_entry = {
                "system": _SYS_PROMPT,
                "user": _unescape(user),
                "assistant": assistant,
            }
            train_data.append(train_entry)
            sample = {
                "inputs": _unescape(user),
                "targets": assistant,
                "system": _SYS_PROMPT,
            }
            messages_data.append(_convert_to_messages(sample))

    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for entry in train_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")

    with open(output_file_messages, "w", encoding="utf-8") as outfile:
        for entry in messages_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


def _knowledge_seed_example_to_test_data(seed_example):
    res = []
    for qna in seed_example["questions_and_answers"]:
        user = qna["question"] + "\n" + seed_example["context"]
        res.append(
            {
                "system": _SYS_PROMPT,
                "user": _unescape(user),
                "assistant": _unescape(qna["answer"]),
            }
        )
    return res


def _gen_test_data(
    leaf_nodes,
    output_file_test,
):
    """
    Generate test data in the format needed by the legacy Linux training
    in instructlab/instructlab.
    """
    test_data = []
    for _, leaf_node in leaf_nodes.items():
        for seed_example in leaf_node:
            if "questions_and_answers" in seed_example:
                test_data.extend(_knowledge_seed_example_to_test_data(seed_example))
                continue

            # skill seed example

            user = seed_example["instruction"]  # question

            if len(seed_example["input"]) > 0:
                user += "\n" + seed_example["input"]  # context

            test_data.append(
                {
                    "system": _SYS_PROMPT,
                    "user": _unescape(user),
                    "assistant": _unescape(seed_example["output"]),  # answer
                }
            )

    with open(output_file_test, "w", encoding="utf-8") as outfile:
        for entry in test_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


def _gen_leaf_node_data(
    leaf_node_data, recipe, output_file_leaf_node, sampling_size=1.0
):
    """
    Write the data generated from each taxonomy leaf node to a file.
    Later on, after all data is generated, the data mixing will read data
    from these files to generate the overall mixed dataset.
    """
    leaf_node_data.to_json(output_file_leaf_node, orient="records", lines=True)
    recipe.add_dataset(output_file_leaf_node, sampling_size)


def _gen_mixed_data(recipe, output_file_mixed, ctx):
    """
    Mix the generated leaf node data into a single dataset and write it to
    disk. The heavy lifting is delegated to the Recipe class.
    """
    if recipe.dataset_added:
        recipe.save_mixed_dataset(
            output_file_mixed,
            ctx.dataset_num_procs,
        )


def _check_pipeline_dir(pipeline):
    for file in ["knowledge.yaml", "freeform_skills.yaml", "grounded_skills.yaml"]:
        if not os.path.exists(os.path.join(pipeline, file)):
            raise GenerateException(
                f"Error: pipeline directory ({pipeline}) does not contain {file}."
            )


def _context_init(
    client: openai.OpenAI,
    model_family: str,
    model_id: str,
    num_instructions_to_generate: int,
    batch_num_workers: Optional[int],
    batch_size: Optional[int],
):
    extra_kwargs = {}
    if batch_size is not None:
        extra_kwargs["batch_size"] = batch_size
        extra_kwargs["batch_num_workers"] = batch_num_workers

    return PipelineContext(
        client=client,
        model_family=model_family,
        model_id=model_id,
        num_instructions_to_generate=num_instructions_to_generate,
        **extra_kwargs,
    )


def _sdg_init(ctx, pipeline):
    pipeline_pkg = None

    # Search for the pipeline in User and Site data directories
    # then for a package defined pipeline
    # and finally pipelines referenced by absolute path
    pd = platformdirs.PlatformDirs(
        appname=os.path.join("instructlab", "sdg"), multipath=True
    )
    for d in pd.iter_data_dirs():
        if os.path.exists(os.path.join(d, pipeline)):
            pipeline = os.path.join(d, pipeline)
            _check_pipeline_dir(pipeline)
            break
    else:
        if pipeline == "full":
            pipeline_pkg = FULL_PIPELINES_PACKAGE
        elif pipeline == "simple":
            pipeline_pkg = SIMPLE_PIPELINES_PACKAGE
        else:
            # Validate that pipeline is a valid directory and that it contains the required files
            if not os.path.exists(pipeline):
                raise GenerateException(
                    f"Error: pipeline directory ({pipeline}) does not exist."
                )
            _check_pipeline_dir(pipeline)

    def load_pipeline(yaml_basename):
        if pipeline_pkg:
            with resources.as_file(
                resources.files(pipeline_pkg).joinpath(yaml_basename)
            ) as yaml_path:
                return Pipeline.from_file(ctx, yaml_path)
        else:
            return Pipeline.from_file(ctx, os.path.join(pipeline, yaml_basename))

    return (
        SDG([load_pipeline("knowledge.yaml")]),
        SDG([load_pipeline("freeform_skills.yaml")]),
        SDG([load_pipeline("grounded_skills.yaml")]),
    )


def _generate_knowledge_qa_dataset(
    logger, generated_dataset: Dataset, keep_context_separate=False
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
        instruction = _get_question(logger, rec)
        response = _get_response(logger, rec)
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


def _create_phase10_ds(logger, generated_dataset: Dataset):
    """
    Create a dataset for Phase 1.0 of downstream training.

    This dataset is in our messages format, with each sample having
    additional context mixed in from other samples to improve the
    training outcomes.
    """
    knowledge_ds = _generate_knowledge_qa_dataset(
        logger, generated_dataset, keep_context_separate=True
    )
    knowledge_ds = _build_raft_dataset(knowledge_ds, p=0.4)

    return knowledge_ds


def _create_phase07_ds(logger, generated_dataset: Dataset):
    """
    Create a dataset for Phase 0.7 of downstream training.

    Phase 0.7 is a pretraining phase, and this dataset contains messages
    with a special `pretraining` role used by downstream training before
    running the full training with the Phase 1.0 dataset.
    """
    # Phase 0.7
    knowledge_ds = _generate_knowledge_qa_dataset(
        logger, generated_dataset, keep_context_separate=False
    )
    knowledge_ds = knowledge_ds.map(_conv_pretrain)

    return knowledge_ds


# This is part of the public API, and used by instructlab.
# TODO - parameter removal needs to be done in sync with a CLI change.
# pylint: disable=unused-argument
def generate_data(
    logger,
    api_base,
    api_key: Optional[str] = None,
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
    output_dir: Optional[str] = None,
    # TODO - not used and should be removed from the CLI
    prompt_file_path: Optional[str] = None,
    # TODO - probably should be removed
    rouge_threshold: Optional[float] = None,
    console_output=True,
    yaml_rules: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    tls_insecure=False,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
    pipeline: Optional[str] = "simple",
    batch_size: Optional[int] = None,
) -> None:
    """Generate data for training and testing a model.

    This currently serves as the primary interface from the `ilab` CLI to the `sdg` library.
    It is somewhat a transitionary measure, as this function existed back when all of the
    functionality was embedded in the CLI. At some stage, we expect to evolve the CLI to
    use the SDG library constructs directly, and this function will likely be removed.

    Args:
        pipeline: This argument may be either an alias defined in a user or site "data directory"
                  or an alias defined by the sdg library ("simple", "full")(if the data directory has no matches),
                  or an absolute path to a directory containing the pipeline YAML files.
                  We expect three files to be present in this directory: "knowledge.yaml",
                    "freeform_skills.yaml", and "grounded_skills.yaml".
    """
    generate_start = time.time()

    # FIXME: remove this when ilab knows to pass batch_size=0 with llama.cpp
    if batch_size is None:
        batch_size = 0

    knowledge_recipe = Recipe(sys_prompt=_SYS_PROMPT)
    skills_recipe = Recipe(sys_prompt=_SYS_PROMPT)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (taxonomy and os.path.exists(taxonomy)):
        raise GenerateException(f"Error: taxonomy ({taxonomy}) does not exist.")

    leaf_nodes = read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules)
    if not leaf_nodes:
        raise GenerateException("Error: No new leaf nodes found in the taxonomy.")

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file_messages = f"messages_{name}_{date_suffix}.jsonl"
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    output_file_train = f"train_{name}_{date_suffix}.jsonl"
    output_file_mixed_knowledge = f"knowledge_train_msgs_{date_suffix}.jsonl"
    output_file_mixed_skills = f"skills_train_msgs_{date_suffix}.jsonl"

    _gen_test_data(
        leaf_nodes,
        os.path.join(output_dir, output_file_test),
    )

    logger.debug(f"Generating to: {os.path.join(output_dir, output_file_test)}")

    orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
    cert = tuple(item for item in orig_cert if item)
    verify = not tls_insecure
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(cert=cert, verify=verify),
    )

    if models.get_model_family(model_family, model_name) == "mixtral":
        model_family = MODEL_FAMILY_MIXTRAL
    else:
        model_family = MODEL_FAMILY_MERLINITE

    ctx = _context_init(
        client,
        model_family,
        model_name,
        num_instructions_to_generate,
        batch_size=batch_size,
        batch_num_workers=num_cpus,
    )

    sdg_knowledge, sdg_freeform_skill, sdg_grounded_skill = _sdg_init(ctx, pipeline)

    mmlu_bench_pipe = mmlubench_pipe_init(ctx)

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    generated_data = None
    for leaf_node in leaf_nodes.values():
        is_knowledge = False
        leaf_node_path = leaf_node[0]["taxonomy_path"].replace("->", "_")
        samples = leaf_node_to_samples(leaf_node, server_ctx_size, chunk_word_count)

        if not samples:
            raise GenerateException("Error: No samples found in leaf node.")

        if samples[0].get("document"):
            sdg = sdg_knowledge
            is_knowledge = True

        elif samples[0].get("seed_context"):
            sdg = sdg_grounded_skill

        else:
            sdg = sdg_freeform_skill

        logger.debug("Samples: %s" % samples)
        ds = Dataset.from_list(samples)
        logger.debug("Dataset: %s" % ds)
        new_generated_data = sdg.generate(ds)
        generated_data = (
            [new_generated_data]
            if generated_data is None
            else generated_data + [new_generated_data]
        )
        logger.info("Generated %d samples" % len(generated_data))
        logger.debug("Generated data: %s" % generated_data)

        if samples[0].get("document"):
            # generate mmlubench data for the current leaf node
            generate_eval_task_data(
                mmlu_bench_pipe,
                leaf_node_path,
                new_generated_data,
                output_dir,
                date_suffix,
            )

        if is_knowledge:
            knowledge_phase_data = _create_phase07_ds(logger, new_generated_data)
            output_file_leaf_knowledge = (
                f"node_datasets_{date_suffix}/{leaf_node_path}_p07.jsonl"
            )
            _gen_leaf_node_data(
                knowledge_phase_data,
                knowledge_recipe,
                os.path.join(output_dir, output_file_leaf_knowledge),
            )

            skills_phase_data = _create_phase10_ds(logger, new_generated_data)
            output_file_leaf_skills = (
                f"node_datasets_{date_suffix}/{leaf_node_path}_p10.jsonl"
            )
            _gen_leaf_node_data(
                skills_phase_data,
                skills_recipe,
                os.path.join(output_dir, output_file_leaf_skills),
            )
        else:
            messages = new_generated_data.map(
                _convert_to_leaf_node_messages,
                fn_kwargs={"logger": logger, "sys_prompt": _SYS_PROMPT},
                num_proc=ctx.dataset_num_procs,
            )
            output_file_leaf = f"node_datasets_{date_suffix}/{leaf_node_path}.jsonl"
            _gen_leaf_node_data(
                messages,
                skills_recipe,
                os.path.join(output_dir, output_file_leaf),
                sampling_size=NUM_SYNTH_SKILLS,
            )

    if generated_data is None:
        generated_data = []

    _gen_train_data(
        logger,
        generated_data,
        os.path.join(output_dir, output_file_train),
        os.path.join(output_dir, output_file_messages),
    )

    _gen_mixed_data(
        knowledge_recipe,
        os.path.join(output_dir, output_file_mixed_knowledge),
        ctx,
    )
    _gen_mixed_data(
        skills_recipe,
        os.path.join(output_dir, output_file_mixed_skills),
        ctx,
    )

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
