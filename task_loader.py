import yaml
import os
def load_tasks(yaml_path: str, diff=None):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        tasks = yaml.safe_load(f)
    if diff:
        return tasks.get(diff, [])
    return tasks

def print_tasks_by_difficulty(tasks: dict):
    difficulties = ['basic', 'easy', 'medium', 'hard', 'complex']
    for difficulty in difficulties:
        print(f"\n=== {difficulty.upper()} ===")
        if difficulty in tasks:
            for i, task in enumerate(tasks[difficulty], start=1):
                print(f"{i:2}. {task}")
                
        else:
            print("  (No tasks)")

def run_basic():
    tasks = load_tasks("tasks.yaml", "basic")
    for i, task in enumerate(tasks, start=1):
        task_name = f"basic_task_{i}"
        print(f"Running task: {task_name}, Description: {task}")
        # os.system(f'python ./simulator/MalmoEnv/run-llm-mem.py  --MEM disable --log {task_name} --userrequest "{task}"')

# python ./simulator/MalmoEnv/run-llm-mem.py  --MEM disable --log test1 --userrequest "make a wooden pickaxe"

if __name__ == "__main__":
    run_basic()
    
    