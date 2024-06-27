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
from instructlab.utils import get_sysprompt
import httpx
import openai

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg import SDG
from instructlab.sdg.default_flows import (
    MMLUBenchFlow,
    SimpleKnowledgeFlow,
    SynthKnowledgeFlow,
)
from instructlab.sdg.pipeline import Pipeline
from instructlab.sdg.taxonomy import leaf_node_to_samples, read_taxonomy_leaf_nodes
from instructlab.sdg.utils import chunking


def _unescape(s):
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
                "system": get_sysprompt(),
                "user": _unescape(user),
                "assistant": _unescape(synth_example["output"]),
            }
        )
    # utils_json.jdump(train_data, output_file_train)
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
    generate_start = time.time()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (taxonomy and os.path.exists(taxonomy)):
        raise SystemExit(f"Error: taxonomy ({taxonomy}) does not exist.")

    leaf_nodes = read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules)

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

    flow_types = []
    if profile == "full":
        flow_types.append(MMLUBenchFlow)
        flow_types.append(SynthKnowledgeFlow)
    elif profile == "simple":
        flow_types.append(SimpleKnowledgeFlow)
    else:
        raise SystemExit(f"Error: profile ({profile}) is not supported.")

    sdg = SDG(
        [
            Pipeline(flow_type(client, model_name, batched).get_flow())
            for flow_type in flow_types
        ]
    )

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    for leaf_node in leaf_nodes.values():
        samples = leaf_node_to_samples(leaf_node)

        # TODO this is broken, just trying to get initial integration to run
        # pylint: disable=consider-using-enumerate
        for i in range(len(samples)):
            samples[i]["document"] = chunking.chunk_document(
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
