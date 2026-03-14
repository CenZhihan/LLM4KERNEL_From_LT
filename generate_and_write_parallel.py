from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os

from config import temperature, top_p
from dataset import dataset
from utils.utils import get_client, get_default_model_from_config

from generate_and_write import (
    generate_prompt,
    generate_and_write_single,
    _agent_output_dir_base,
)
from Agent_kernel.agent_config import AgentToolMode
from Agent_kernel.agent_runner import KernelTask, AgentResult, generate_kernel_with_agent


def _generate_one_direct(op, out_dir, language, strategy, model):
    """Direct LLM: 单算子任务。"""
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[INFO] Already generated at {out_dir}/{op}.txt, skip")
        return op, None
    try:
        print(f"[INFO] Generate kernel for op {op}, strategy is {strategy}")
        prompt = generate_prompt(language, strategy, op)
        client = get_client(model)
        generate_and_write_single(prompt, client, out_dir, op, model)
        print(f"[INFO] Done op {op}")
        return op, None
    except Exception as e:
        print(f"[FAIL] op {op}: {e}")
        return op, e


def _generate_one_agent(
    op,
    language,
    base_strategy,
    tool_mode: AgentToolMode,
    model_name: str,
    run: int,
):
    out_dir_base = _agent_output_dir_base(
        language, base_strategy, model_name, tool_mode
    )
    out_dir = os.path.join(out_dir_base, f"run{run}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{op}.txt")
    if os.path.exists(out_path):
        print(f"[INFO][Agent] Already generated at {out_path}, skip")
        return op, None
    try:
        print(
            f"[INFO][Agent] Generate kernel for op {op}, "
            f"base_strategy={base_strategy}, tools={tool_mode.value}"
        )
        task = KernelTask(language=language, op=op, strategy_name=base_strategy)
        result: AgentResult = generate_kernel_with_agent(
            task=task, tool_mode=tool_mode
        )
        with open(out_path, "w") as f:
            f.write(result.raw_answer)
        print(f"[INFO][Agent] Done op {op}")
        return op, None
    except Exception as e:
        print(f"[FAIL][Agent] op {op}: {e}")
        return op, e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model evaluation with specified parameters (parallel)."
    )
    parser.add_argument("--runs", type=int, default=1, help="Number of runs.")
    _default_model = get_default_model_from_config() or "deepseek-chat"
    parser.add_argument("--model", type=str, default=_default_model, help="Model name.")
    parser.add_argument("--language", type=str, default="cuda", help="Language to use.")
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
        "--workers",
        type=int,
        default=4,
        help="Max concurrent API calls (default 4).",
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

    args = parser.parse_args()
    runs = args.runs
    model = args.model
    language = args.language
    strategy = args.strategy
    categories = args.categories
    workers = args.workers
    use_agent = args.use_agent
    agent_tools = AgentToolMode(args.agent_tools)

    print(f"Runs: {runs}")
    print(f"Model: {model}")
    print(f"Language: {language}")
    print(f"Strategy: {strategy}")
    print(f"Categories: {categories}")
    print(f"Workers: {workers}")
    print(f"Use Agent: {use_agent}")
    if use_agent:
        print(f"Agent tools: {agent_tools.value}")

    op_tested = list(dataset.keys())
    if categories != ["all"]:
        op_tested = [op for op in op_tested if dataset[op]["category"] in categories]

    if "/" in model:
        model_name = model.split("/")[1]
    else:
        model_name = model

    for run in range(runs):
        if not use_agent:
            out_dir = f"output/{language}/{strategy}/{temperature}-{top_p}/{model_name}/run{run}"
            os.makedirs(out_dir, exist_ok=True)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _generate_one_direct, op, out_dir, language, strategy, model
                    ): op
                    for op in op_tested
                }
                for future in as_completed(futures):
                    op, err = future.result()
                    if err is not None:
                        pass
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _generate_one_agent,
                        op,
                        language,
                        strategy,
                        agent_tools,
                        model_name,
                        run,
                    ): op
                    for op in op_tested
                }
                for future in as_completed(futures):
                    op, err = future.result()
                    if err is not None:
                        pass
