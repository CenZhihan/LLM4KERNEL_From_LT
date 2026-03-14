# Using LLM to generate code and output it to file
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import importlib
import os
from typing import List, Tuple

from config import temperature, num_completions, max_tokens, top_p
from dataset import dataset
from prompt_generators.prompt_registry import PROMPT_REGISTRY
from utils.utils import get_client, get_default_model_from_config

from Agent_kernel.agent_config import AgentToolMode
from Agent_kernel.agent_runner import KernelTask, AgentResult, generate_kernel_with_agent


def generate_and_write_single(prompt, client, out_dir, op, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=temperature,
        n=num_completions,
        # max_tokens=max_tokens,
        top_p=top_p,
    )
    reasoning_content = ""  # 完整思考过程
    answer_content = ""  # 完整回复
    is_answering = False  # 是否进入回复阶段
    for chunk in response:
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            reasoning_content += delta.reasoning_content
        if hasattr(delta, "content") and delta.content:
            if not is_answering:
                is_answering = True
            answer_content += delta.content
    if reasoning_content != "":
        with open(os.path.join(out_dir, f"{op}_cot.txt"), "w") as out_file:
            out_file.write(reasoning_content)
    with open(os.path.join(out_dir, f"{op}.txt"), "w") as out_file:
        out_file.write(answer_content)


def generate_prompt(language, strategy_name, op):
    if language not in PROMPT_REGISTRY or strategy_name not in PROMPT_REGISTRY[language]:
        try:
            importlib.import_module(f"prompt_generators.{language}_{strategy_name}")
        except ImportError as e:
            raise ValueError(
                f"Unsupported language/platform: {language} (module not found)"
            ) from e

    strategy = PROMPT_REGISTRY[language][strategy_name]
    return strategy.generate(op)


def _agent_output_dir_base(
    language: str,
    base_strategy: str,
    model_name: str,
    tool_mode: AgentToolMode,
) -> str:
    tools_name = {
        AgentToolMode.NO_TOOL: "no_tool",
        AgentToolMode.KB_ONLY: "kb_only",
        AgentToolMode.WEB_ONLY: "web_only",
        AgentToolMode.KB_AND_WEB: "kb_and_web",
    }[tool_mode]
    strategy_dir = f"agent_{base_strategy}_tools={tools_name}"
    return f"output/{language}/{strategy_dir}/{temperature}-{top_p}/{model_name}"


def generate_and_write_direct(out_dir, language, model, op_tested, strategy):
    client = get_client(model)
    for op in op_tested:
        print(f"[INFO] Generate kernel for op {op}, strategy is {strategy}")
        prompt = generate_prompt(language, strategy, op)
        if os.path.exists(os.path.join(out_dir, f"{op}.txt")):
            print(f"[INFO] Already generated at {out_dir}/{op}.txt, skip")
            continue
        generate_and_write_single(prompt, client, out_dir, op, model)


def _agent_worker(
    language: str,
    base_strategy: str,
    tool_mode: AgentToolMode,
    model_name: str,
    run: int,
    op: str,
) -> Tuple[str, bool]:
    out_dir_base = _agent_output_dir_base(
        language, base_strategy, model_name, tool_mode
    )
    out_dir = os.path.join(out_dir_base, f"run{run}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[INFO][Agent] Already generated at {out_path}, skip")
        return op, False

    print(
        f"[INFO][Agent] Generate kernel for op {op}, "
        f"base_strategy={base_strategy}, "
        f"tools={tool_mode.value}"
    )

    task = KernelTask(language=language, op=op, strategy_name=base_strategy)
    result: AgentResult = generate_kernel_with_agent(
        task=task, tool_mode=tool_mode
    )

    with open(out_path, "w") as out_file:
        out_file.write(result.raw_answer)
    return op, True


def generate_and_write_agent(
    language: str,
    model_name: str,
    op_tested: List[str],
    base_strategy: str,
    tool_mode: AgentToolMode,
    runs: int,
    max_workers: int,
):
    for run in range(runs):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _agent_worker,
                    language,
                    base_strategy,
                    tool_mode,
                    model_name,
                    run,
                    op,
                )
                for op in op_tested
            ]
            for fut in as_completed(futures):
                try:
                    op, written = fut.result()
                    if written:
                        print(f"[INFO][Agent] Wrote result for op {op}")
                except Exception as e:
                    print(f"[ERROR][Agent] Failed to generate op with Agent: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model evaluation with specified parameters."
    )

    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    _default_model = get_default_model_from_config() or "deepseek-chat"
    parser.add_argument(
        "--model",
        type=str,
        default=_default_model,
        help=(
            "Model name (default: from api_config if USE_API_CONFIG=1, "
            "else deepseek-chat)."
        ),
    )
    parser.add_argument(
        "--language", type=str, default="cuda", help="Language to use."
    )
    parser.add_argument(
        "--strategy", type=str, default="add_shot", help="Strategy type."
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["activation"],
        help="List of categories.",
    )
    parser.add_argument(
        "--use_agent",
        action="store_true",
        help="Use LangGraph Agent to generate kernels.",
    )
    parser.add_argument(
        "--agent_tools",
        type=str,
        default="no_tool",
        choices=[m.value for m in AgentToolMode],
        help="Tool mode for Agent (no_tool/kb_only/web_only/kb_and_web).",
    )
    parser.add_argument(
        "--agent_workers",
        type=int,
        default=4,
        help="Max parallel workers for Agent generation.",
    )

    args = parser.parse_args()

    runs = args.runs
    model = args.model
    language = args.language
    strategy = args.strategy
    categories = args.categories
    use_agent = args.use_agent
    agent_tools = AgentToolMode(args.agent_tools)
    agent_workers = args.agent_workers

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")
    print(f"Use Agent: {use_agent}")
    if use_agent:
        print(f"Agent tools: {agent_tools.value}")
        print(f"Agent workers: {agent_workers}")

    op_tested = list(dataset.keys())
    if categories != ["all"]:
        op_tested = [op for op in op_tested if dataset[op]["category"] in categories]

    if "/" in model:
        model_name = model.split("/")[1]
    else:
        model_name = model

    if not use_agent:
        for run in range(runs):
            out_dir = (
                f"output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}"
            )
            os.makedirs(out_dir, exist_ok=True)
            generate_and_write_direct(out_dir, language, model, op_tested, strategy)
    else:
        generate_and_write_agent(
            language=language,
            model_name=model_name,
            op_tested=op_tested,
            base_strategy=strategy,
            tool_mode=agent_tools,
            runs=runs,
            max_workers=agent_workers,
        )

