import yaml
import os
import time

def load_tasks(yaml_path: str, diff=None):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        tasks = yaml.safe_load(f)
    if diff:
        return tasks.get(diff, [])
    return tasks


def run_basic_LLM(task_file="tasks.yaml"):
    tasks = load_tasks(task_file, "basic")

    print("Running basic LLM tasks:")
    


# python ./simulator/MalmoEnv/run-llm-mem.py  --MEM disable --log test2 --userrequest "mine log"

if __name__ == "__main__":
    run_basic_LLM()
    
    