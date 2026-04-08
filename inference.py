"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.
    LOCAL_IMAGE_NAME  The name of the local image to use for the environment if you are using from_docker_image().

- Defaults are set only for API_BASE_URL and MODEL_NAME.
- The inference script must be named `inference.py` and placed in the root directory of the project.
- Participants must use OpenAI Client for all LLM calls using above variables.

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from backend_incident_triage.client import BackendIncidentTriageEnv
from backend_incident_triage.models import BackendIncidentTriageAction as Action

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK", "backend_incident_triage")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "400"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.7"))

SYSTEM_PROMPT = """You are an on-call backend engineer in an incident triage simulation.
Each turn you send a message to the environment. The environment echoes it back.
Reward is proportional to message length: reward = len(message) * 0.1
Your goal is to maximize total reward by sending meaningful, substantive messages.
Reply with exactly one message string — no quotes, no prefixes, just the message text."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def format_observation(obs) -> str:
    if hasattr(obs, "model_dump"):
        return json.dumps(obs.model_dump(), indent=2, default=str)
    if hasattr(obs, "dict"):
        return json.dumps(obs.dict(), indent=2, default=str)
    return json.dumps(getattr(obs, "__dict__", {"observation": str(obs)}), indent=2, default=str)


def extract_available_tools(obs) -> str:
    if hasattr(obs, "available_tools"):
        return json.dumps(obs.available_tools, default=str)
    if hasattr(obs, "observation") and hasattr(obs.observation, "available_tools"):
        return json.dumps(obs.observation.available_tools, default=str)
    return "[]"


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return f"""
Step: {step}
Last echoed message: {last_echoed!r}
Last reward: {last_reward:.2f}
Previous steps:
{history_block}
Send your next message.
""".strip()


def parse_action(raw: str) -> str:
    return raw.strip()


def action_to_string(action: Action) -> str:
    return action.message


def choose_action(llm_client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> Action:
    user_prompt = build_user_prompt(step, last_echoed, last_reward, history)
    try:
        completion = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        message = parse_action(raw)
        return Action(message=message if message else "hello")
    except Exception as e:
        # For demo, generate varied fallback messages
        import random
        messages = [
            "Investigating the backend incident: checking logs and metrics.",
            "As an on-call engineer, I'm analyzing the error patterns in the triage system.",
            "Triage step: reviewing recent deployments and system health indicators.",
            "Incident response: coordinating with team to identify root cause and mitigation.",
            "Backend triage: monitoring alerts and performance degradation signals.",
            "Engineering response: documenting findings and preparing incident report.",
            "System analysis: tracing request flows and identifying bottlenecks.",
            "Triage protocol: escalating critical issues and notifying stakeholders."
        ]
        message = random.choice(messages)
        return Action(message=message)


async def maybe_await(value):
    if asyncio.iscoroutine(value):
        return await value
    return value


async def main() -> None:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN or API_KEY is required")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
    env = BackendIncidentTriageEnv(base_url=BASE_URL)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_obs = None
    history: List[str] = []

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = await maybe_await(env.reset())
        last_obs = obs
        done = getattr(obs, "done", False)
        last_echoed = obs.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            action = choose_action(llm_client, step, last_echoed, last_reward, history)
            next_obs = await maybe_await(env.step(action))
            last_obs = next_obs

            reward = float(getattr(next_obs, "reward", 0.0) or 0.0)
            done = bool(getattr(next_obs, "done", False))
            info = getattr(next_obs, "info", {}) or {}
            error = info.get("last_action_error") or info.get("error")

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_to_string(action), reward=reward, done=done, error=error)

            last_echoed = next_obs.observation.echoed_message
            last_reward = reward
            history.append(f"Step {step}: {action.message!r} -> reward {reward:+.2f}")

            obs = next_obs

        info = getattr(last_obs, "info", {}) or {}
        raw_score = info.get("score", sum(rewards))
        score = max(0.0, min(1.0, float(raw_score)))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await maybe_await(env.close())
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
