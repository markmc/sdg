# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import os
import re
import time

# Third Party
# instructlab - All of these need to go away (other than sdg) - issue #6
from datasets import Dataset
import httpx
import instructlab.utils
import openai

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg import SDG, utils
from instructlab.sdg.default_flows import (
    MMLUBenchFlow,
    SimpleKnowledgeFlow,
    SynthKnowledgeFlow,
)
from instructlab.sdg.pipeline import Pipeline

_WORD_DENYLIST = [
    "image",
    "images",
    "graph",
    "graphs",
    "picture",
    "pictures",
    "file",
    "files",
    "map",
    "maps",
    "draw",
    "plot",
    "go to",
    "video",
    "audio",
    "music",
    "flowchart",
    "diagram",
]


def writeline2file(logfile, line):
    t = datetime.now().replace(microsecond=0).isoformat()
    with open(logfile, "a", encoding="utf-8") as fp:
        fp.write(f"{t} - {line}\n")


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def read_taxonomy(*args, **kwargs):
    return instructlab.utils.read_taxonomy(*args, **kwargs)


def unescape(s):
    return bytes(s, "utf-8").decode("utf-8")


def _gen_train_data(logger, machine_instruction_data, output_file_train):
    train_data = []
    for synth_example in machine_instruction_data:
        logger.debug(synth_example)
        user = synth_example.get("instruction", "")
        if len(synth_example.get("input", "")) > 0:
            user += "\n" + synth_example["input"]
        train_data.append(
            {
                "system": utils.get_sysprompt(),
                "user": unescape(user),
                "assistant": unescape(synth_example["output"]),
            }
        )
    # utils.jdump(train_data, output_file_train)
    with open(output_file_train, "w", encoding="utf-8") as outfile:
        for entry in train_data:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write("\n")


# TODO - parameter removal needs to be done in sync with a CLI change.
# pylint: disable=unused-argument
def generate_data(
    logger,
    api_base,
    tls_insecure,
    # TODO - not yet used. Right now the lib will guess based on the model name
    # but we should pass this along if specified
    model_family: str,
    yaml_rules: Optional[str] = None,
    output_dir: Optional[str] = None,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
    # TODO - not used and should be removed from the CLI
    prompt_file_path: Optional[str] = None,
    model_name: Optional[str] = None,
    # TODO - not used -- when batching is enabled, this is relevant.
    # Right now the code hard codes 8 cpus for batching
    num_cpus: Optional[int] = None,
    # TODO - not yet used, but should be presumably
    num_instructions_to_generate: Optional[int] = None,
    # TODO - not used, can probably be removed
    num_prompt_instructions=2,
    # TODO - determine if this is relevant
    request_batch_size=5,
    # TODO - probably should be removed
    temperature=1.0,
    # TODO - probably should be removed
    top_p=1.0,
    # TODO - probably should be removed
    rouge_threshold: Optional[float] = None,
    console_output=True,
    api_key: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
    # TODO need to update the CLI to specify which profile to use (simple or full at the moment)
    profile: Optional[str] = "simple",
):
    seed_instruction_data = []
    generate_start = time.time()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # check taxonomy first then seed_tasks_path
    # throw an error if both not found
    # pylint: disable=broad-exception-caught,raise-missing-from
    if taxonomy and os.path.exists(taxonomy):
        # TODO -- rewrite how this returns data so we don't have to do
        # so much transformation on it
        seed_instruction_data = read_taxonomy(
            logger, taxonomy, taxonomy_base, yaml_rules
        )
    else:
        raise SystemExit(f"Error: taxonomy ({taxonomy}) does not exist.")

    # Transform into a more convenient format to feed into our updated SDG library
    leaf_nodes = {}
    for seed in seed_instruction_data:
        node = leaf_nodes.setdefault(seed["taxonomy_path"], [])
        node.append(seed)
        leaf_nodes[seed["taxonomy_path"]] = node

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file = f"generated_{name}_{date_suffix}.json"
    logger.debug(f"Generating to: {os.path.join(output_dir, output_file)}")

    orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
    cert = tuple(item for item in orig_cert if item)
    verify = not tls_insecure
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(cert=cert, verify=verify),
    )

    # TODO -- llama-cpp doesn't support batching, we need to get a hint from the CLI
    # about whether we can turn this on (whether vllm is used or not)
    batched = False

    sdg = None
    if profile == "full":
        mmlu_flow = MMLUBenchFlow(client, model_name, batched).get_flow()
        mmlu_pipe = Pipeline(mmlu_flow)
        knowledge_flow = SynthKnowledgeFlow(client, model_name, batched).get_flow()
        knowledge_pipe = Pipeline(knowledge_flow)
        sdg = SDG([mmlu_pipe, knowledge_pipe])
    elif profile == "simple":
        knowledge_flow = SimpleKnowledgeFlow(client, model_name, batched).get_flow()
        sdg = SDG([Pipeline(knowledge_flow)])
    else:
        raise SystemExit(f"Error: profile ({profile}) is not supported.")

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    for leaf_node in leaf_nodes.values():
        # TODO -- only handles knowledge leaf nodes, need to add support for other types
        if leaf_node[0].get("document") is None:
            logger.error("Only knowledge leaf nodes supported at the moment")
            continue

        samples = [{}]
        # pylint: disable=consider-using-enumerate
        for i in range(len(leaf_node)):
            samples[-1].setdefault("task_description", leaf_node[i]["task_description"])
            samples[-1].setdefault("document", leaf_node[i]["document"])
            # TODO - fix read_taxonomy() to return the domain. It's not included right now.
            samples[-1].setdefault("domain", leaf_node[i].get("domain", "general"))
            if "question_3" in samples[-1]:
                samples.append({})
            if "question_1" not in samples[-1]:
                samples[-1]["question_1"] = leaf_node[i]["instruction"]
                samples[-1]["response_1"] = leaf_node[i]["output"]
            elif "question_2" not in samples[-1]:
                samples[-1]["question_2"] = leaf_node[i]["instruction"]
                samples[-1]["response_2"] = leaf_node[i]["output"]
            else:
                samples[-1]["question_3"] = leaf_node[i]["instruction"]
                samples[-1]["response_3"] = leaf_node[i]["output"]
        # wrap back around to the beginning if the number of examples was not
        # evenly divisble by 3
        if "question_2" not in samples[-1]:
            samples[-1]["question_2"] = leaf_node[0]["instruction"]
            samples[-1]["response_2"] = leaf_node[0]["output"]
        if "question_3" not in samples[-1]:
            samples[-1]["question_3"] = leaf_node[1 if len(leaf_node) > 1 else 0][
                "instruction"
            ]
            samples[-1]["response_3"] = leaf_node[1 if len(leaf_node) > 1 else 0][
                "output"
            ]

        # TODO this is broken, just trying to get initial integration to run
        # pylint: disable=consider-using-enumerate
        for i in range(len(samples)):
            samples[i]["document"] = utils.chunk_document(
                documents=samples[i]["document"],
                server_ctx_size=server_ctx_size,
                chunk_word_count=chunk_word_count,
            )[0]

        # TODO -- there is a parameter for how many samples to generate, but we ignore it so far

        ds = Dataset.from_list(samples)
        generated_data = sdg.generate(ds)
        logger.info("Generated %d samples" % len(generated_data))
        logger.debug("Generated data: %s" % generated_data)

        _gen_train_data(logger, generated_data, os.path.join(output_dir, output_file))

    generate_duration = time.time() - generate_start
    logger.info(f"Generation took {generate_duration:.2f}s")
