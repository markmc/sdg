# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from importlib import resources
import operator
import os

# First Party
from instructlab.sdg.utils import models

# Local
from .filterblock import FilterByValueBlock
from .iterblock import IterBlock
from .llmblock import LLMBlock

MODEL_PROMPT_MIXTRAL = "<s> [INST] {prompt} [/INST]"
MODEL_PROMPT_MERLINITE = "'<|system|>\nYou are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.\n<|user|>\n{prompt}\n<|assistant|>\n'"


def _get_model_prompt(model_id):
    return (
        MODEL_PROMPT_MIXTRAL
        if models.get_model_family(None, model_id) == "mixtral"
        else MODEL_PROMPT_MERLINITE
    )


class Flow(ABC):
    def __init__(self, client, model_id, batched=True) -> None:
        self.client = client
        self.model_id = model_id
        self.batched = batched

    @abstractmethod
    def get_flow(self) -> list:
        pass


class SimpleKnowledgeFlow(Flow):
    def get_flow(self) -> list:
        sdg_base = resources.files(__package__)
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_knowledge",
                    "config_path": os.path.join(
                        sdg_base, "configs/knowledge/simple_generate_qa.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["output"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
                "drop_duplicates": ["output"],
            },
        ]


class MMLUBenchFlow(Flow):
    def get_flow(self) -> list:
        sdg_base = resources.files(__package__)
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_mmlu_knowledge",
                    "config_path": os.path.join(
                        sdg_base, "configs/knowledge/mcq_generation.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["mmlubench_question", "mmlubench_answer"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "gen_kwargs": {
                    "temperature": 0,
                    "max_tokens": 2048,
                },
                "drop_duplicates": ["mmlubench_question"],
            },
        ]


class SynthKnowledgeFlow(Flow):
    def get_flow(self) -> list:
        sdg_base = resources.files(__package__)
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_knowledge",
                    "config_path": os.path.join(
                        sdg_base, "configs/knowledge/generate_questions_responses.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["question", "response"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_faithfulness_qa_pair",
                    "config_path": os.path.join(
                        sdg_base, "configs/knowledge/evaluate_faithfulness.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["explanation", "judgment"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_faithfulness",
                    "filter_column": "judgment",
                    "filter_value": "YES",
                    "operation": operator.eq,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["judgment", "explanation"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_relevancy_qa_pair",
                    "config_path": os.path.join(
                        sdg_base, "configs/knowledge/evaluate_relevancy.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["feedback", "score"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_relevancy",
                    "filter_column": "score",
                    "filter_value": "2",
                    "operation": operator.eq,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["feedback", "score"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_verify_question",
                    "config_path": os.path.join(
                        sdg_base, "configs/knowledge/evaluate_question.yaml"
                    ),
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["explanation", "rating"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "gen_kwargs": {
                    "max_tokens": 2048,
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_verify_question",
                    "filter_column": "rating",
                    "filter_value": "1",
                    "operation": operator.eq,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["explanation", "rating", "__index_level_0__"],
            },
        ]


class SynthSkillsFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_questions",
                    "config_path": "src/instructlab/sdg/configs/skills/freeform_questions.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["question"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "num_samples": 30,
                        "batched": self.batched,
                    },
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_questions",
                    "config_path": "src/instructlab/sdg/configs/skills/evaluate_freeform_questions.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["evaluation", "score"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_questions",
                    "filter_column": "score",
                    "filter_value": 1,
                    "operation": operator.eq,
                    "convert_dtype": int,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["evaluation", "score", "num_samples"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_responses",
                    "config_path": "src/instructlab/sdg/configs/skills/freeform_responses.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["answer"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "evaluate_qa_pair",
                    "config_path": "src/instructlab/sdg/configs/skills/evaluate_freeform_pair.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["evaluation", "score"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_qa_pair",
                    "filter_column": "score",
                    "filter_value": 2,
                    "operation": operator.ge,
                    "convert_dtype": int,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["evaluation", "score"],
            },
        ]


class SynthGroundedSkillsFlow(Flow):
    def get_flow(self) -> list:
        return [
            {
                "block_type": IterBlock,
                "block_config": {
                    "block_name": "context_iter",
                    "num_iters": 10,
                    "block_type": LLMBlock,
                    "block_kwargs": {
                        "block_name": "gen_contexts",
                        "config_path": "src/instructlab/sdg/configs/skills/contexts.yaml",
                        "client": self.client,
                        "model_id": self.model_id,
                        "model_prompt": _get_model_prompt(self.model_id),
                        "output_cols": ["context"],
                        "batch_kwargs": {
                            "num_procs": 8,
                            "batched": self.batched,
                        },
                    },
                    "gen_kwargs": {
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    },
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_grounded_questions",
                    "config_path": "src/instructlab/sdg/configs/skills/grounded_questions.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["question"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
                "drop_duplicates": ["question"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "eval_grounded_questions",
                    "config_path": "src/instructlab/sdg/configs/skills/evaluate_grounded_questions.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["evaluation", "score"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_grounded_questions",
                    "filter_column": "score",
                    "filter_value": 1,
                    "operation": operator.eq,
                    "convert_dtype": int,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
                "drop_columns": ["evaluation", "score", "num_samples"],
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "gen_grounded_responses",
                    "config_path": "src/instructlab/sdg/configs/skills/grounded_responses.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["answer"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
            },
            {
                "block_type": LLMBlock,
                "block_config": {
                    "block_name": "evaluate_grounded_qa_pair",
                    "config_path": "src/instructlab/sdg/configs/skills/evaluate_grounded_pair.yaml",
                    "client": self.client,
                    "model_id": self.model_id,
                    "model_prompt": _get_model_prompt(self.model_id),
                    "output_cols": ["evaluation", "score"],
                    "batch_kwargs": {
                        "num_procs": 8,
                        "batched": self.batched,
                    },
                },
            },
            {
                "block_type": FilterByValueBlock,
                "block_config": {
                    "block_name": "filter_grounded_qa_pair",
                    "filter_column": "score",
                    "filter_value": 2,
                    "operation": operator.ge,
                    "convert_dtype": int,
                    "batch_kwargs": {
                        "num_procs": 8,
                    },
                },
            },
        ]
