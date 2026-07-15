import argparse
import asyncio
from pathlib import Path

from ebs.runtime.agents import OrchestraAgent, SimpleAgent, WorkforceAgent, get_agent
from ebs.runtime.agents.orchestra import OrchestraStreamEvent
from ebs.runtime.config import ConfigLoader
from ebs.runtime.utils.agents_utils import AgentsUtils
from ebs.runtime.utils.print_utils import PrintUtils


def normalize_agent_config_name(config: str) -> str:
    """Normalize user-facing config paths to the format expected by ConfigLoader."""
    normalized = config.strip().replace("\\", "/")
    if normalized.endswith(".yaml"):
        normalized = normalized[:-5]
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("configs/"):
        normalized = normalized[len("configs/") :]
    if normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive CLI chat for EBS agents.")
    parser.add_argument(
        "--config",
        type=str,
        default="simple/base.yaml",
        help="Agent config name or path, e.g. simple/base.yaml or configs/agents/simple/base.yaml.",
    )
    parser.add_argument("--query", type=str, default="", help="Run a single query and exit.")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streamed output for simple/orchestra agents.",
    )
    return parser.parse_args()


async def print_orchestra_stream_events(stream) -> None:
    async for event in stream.stream_events():
        if isinstance(event, OrchestraStreamEvent):
            if event.name == "plan_start":
                PrintUtils.print_info("[planner] creating plan...", color="cyan")
            elif event.name == "plan" and event.item is not None:
                PrintUtils.print_info("[planner] plan ready:", color="cyan")
                for i, subtask in enumerate(event.item.todo, 1):
                    PrintUtils.print_info(f"  {i}. {subtask.task} ({subtask.agent_name})", color="cyan")
            elif event.name == "worker" and event.item is not None:
                PrintUtils.print_info(f"[worker] {event.item.task}", color="green")
                if event.item.output:
                    PrintUtils.print_bot(event.item.output)
            elif event.name == "report_start":
                PrintUtils.print_info("[reporter] writing final answer...", color="cyan")
            elif event.name == "report" and event.item is not None:
                PrintUtils.print_bot(event.item.output)
        else:
            await AgentsUtils.print_stream_events(_single_event_iterator(event))


async def _single_event_iterator(event):
    yield event


async def run_once(agent: SimpleAgent | OrchestraAgent | WorkforceAgent, query: str, stream: bool) -> None:
    if isinstance(agent, SimpleAgent):
        await agent.build()
        if stream:
            run_result_streaming = agent.run_streamed(query)
            await AgentsUtils.print_stream_events(run_result_streaming.stream_events())
            agent.input_items = run_result_streaming.to_input_list()
            agent.current_agent = run_result_streaming.last_agent
        else:
            await agent.chat(query)
        return

    if isinstance(agent, OrchestraAgent):
        if stream:
            run_result_streaming = agent.run_streamed(query)
            await print_orchestra_stream_events(run_result_streaming)
        else:
            task_recorder = await agent.run(query)
            PrintUtils.print_bot(task_recorder.final_output)
        return

    if isinstance(agent, WorkforceAgent):
        task_recorder = await agent.run(query)
        PrintUtils.print_bot(task_recorder.final_output)
        return

    raise ValueError(f"Unsupported agent type: {type(agent).__name__}")


async def interactive_chat(agent: SimpleAgent | OrchestraAgent | WorkforceAgent, stream: bool) -> None:
    PrintUtils.print_info("Type a message to chat. Use /clear to reset history, /exit to quit.", color="gray")
    while True:
        try:
            user_input = (await PrintUtils.async_print_input("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            PrintUtils.print_info("\nBye.", color="gray")
            break

        if not user_input:
            continue
        if user_input.lower() in {"/exit", "exit", "quit", "/quit"}:
            PrintUtils.print_info("Bye.", color="gray")
            break
        if user_input.lower() == "/clear":
            if isinstance(agent, SimpleAgent):
                agent.clear_input_items()
                PrintUtils.print_info("Chat history cleared.", color="gray")
            else:
                PrintUtils.print_info("This agent type does not keep CLI chat history.", color="gray")
            continue

        await run_once(agent, user_input, stream=stream)


async def main_async() -> None:
    args = parse_args()
    config_name = normalize_agent_config_name(args.config)
    config = ConfigLoader.load_agent_config(config_name)
    agent = get_agent(config)

    PrintUtils.print_info(f"Loaded config: {config_name}", color="gray")
    PrintUtils.print_info(f"Agent type: {config.type}", color="gray")

    if args.query:
        await run_once(agent, args.query, stream=not args.no_stream)
        return

    await interactive_chat(agent, stream=not args.no_stream)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
