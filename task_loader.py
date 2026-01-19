import yaml
import os
import time


def load_tasks(yaml_path: str, diff=None):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 遍历所有难度
    # for difficulty in ['basic', 'easy', 'medium', 'hard', 'complex']:
    #     if difficulty not in data:
    #         continue
    #     print(f"\n{'='*20} {difficulty.upper()} {'='*20}")
    #     tasks = data[difficulty]
    #     for idx, task in enumerate(tasks, start=1):
    #         print(f"\nTask {idx}:")
    #         print(f"  Name:        {task['task_name']}")
    #         print(f"  Description: {task['description']}")
    #         print(f"  Steps:       {task['steps']}")
    #         print(f"  Judge:       {task['judge']}")

    if diff:
        return data.get(diff, [])
    return data

def run_LLM(task_file="tasks.yaml", task_level="basic", times = 3, arg=""):
    tasks = load_tasks(task_file, task_level)
    print(f"Running {task_level} LLM tasks:")
    for i in range(times):  # 运行3次
        for task in tasks:
            print(f"Task Name: {task['task_name']}")
            print(f"Description: {task['description']}")
            print(f"Steps: {task['steps']}")
            print(f"Judge: {task['judge']}")
            print("-" * 40)
            task_name = task['task_name'].replace(" ", "_")
            task_description = task['description']
            task_judge = task['judge']
            task_steps = task['steps']
            mmyydd = time.strftime("%y%m%d", time.localtime())
            hhmm = time.strftime("%H%M", time.localtime())

            
            log_name = f"{task_level}_{task_name}_LLM_{i}_{mmyydd}_{hhmm}"
            run_cmd = f'''python ./simulator/MalmoEnv/run-llm-mem.py  \
                        --MEM disable \
                        --log {log_name} \
                        --userrequest "{task_description}" \
                        --check "{task_judge}" \
                        --subSteps {task_steps} \
                        --mission simulator/MalmoEnv/missions/world1.xml'''
            if arg != "":
                run_cmd += f"\
                    {arg} "
                    
            print(f"Running command:\n{run_cmd}")
            os.system(run_cmd)

            print(f"Completed run {i+1} for task '{task_name}'.")
            print("=" * 60)
        
def run_MEM(task_file="tasks.yaml", task_level="basic", times = 3, arg=""):
    tasks = load_tasks(task_file, task_level)
    print(f"Running {task_level} MEM tasks:")
    for i in range(times):  # 运行3次
        for task in tasks:
            print(f"Task Name: {task['task_name']}")
            print(f"Description: {task['description']}")
            print(f"Steps: {task['steps']}")
            print(f"Judge: {task['judge']}")
            print("-" * 40)
            task_name = task['task_name'].replace(" ", "_")
            task_description = task['description']
            task_judge = task['judge']
            task_steps = task['steps']
            mmyydd = time.strftime("%y%m%d", time.localtime())
            hhmm = time.strftime("%H%M", time.localtime())

            
            log_name = f"{task_level}_{task_name}_MEM_{i}_{mmyydd}_{hhmm}"
            run_cmd = f'''python ./simulator/MalmoEnv/run-llm-mem.py  \
                        --MEM enable \
                        --log {log_name} \
                        --userrequest "{task_description}" \
                        --check "{task_judge}" \
                        --subSteps {task_steps} \
                        --mission simulator/MalmoEnv/missions/world1.xml'''
            if arg != "":
                run_cmd += f"\
                    {arg} "

            print(f"Running command:\n{run_cmd}")
            os.system(run_cmd)
                
            print(f"Completed run {i+1} for task '{task_name}'.")
            print("=" * 60)

        
'''
python ./simulator/MalmoEnv/run-llm-mem.py  \
    --MEM disable \
    --log {task_level}/{task_name}/{mmyydd}_LLM \
    --userrequest {task_description} \
    --check {task_judge}\

python ./simulator/MalmoEnv/run-llm-mem.py  \
    --MEM disable \
    --log test2 \
    --userrequest "mine log" 
    --check "log"\

python ./simulator/MalmoEnv/run-study.py --mission simulator/MalmoEnv/missions/world4.xml 
'''

'''
Running command:
python ./simulator/MalmoEnv/run-llm-mem.py

--MEM enable
--log basic_craft_crafting_table_MEM_2_251220_1956
--userrequest "craft one crafting table from wooden plank"
--check "crafting_table"
--subSteps 100
--mission simulator/MalmoEnv/missions/world1.xml

'''

if __name__ == "__main__":
    # 12.30 00:00 - 15:06 不分解子任务的 消融 + 短期，长期 （混合-最终的）消融 不知道跑到哪个任务了
    
    # 不分解子任务的 消融
    # run_LLM(task_level="basic", times = 5)
    # run_MEM(task_level="basic", times = 5)
    # run_LLM(task_level="easy", times = 5)
    # run_MEM(task_level="easy", times = 5)
    
    # 短期，长期 （混合-最终的）消融
    # run_MEM(task_level="basic", times = 5, arg = "--scenefile scene_info-tem.json")
    # run_MEM(task_level="easy", times = 5, arg = "--scenefile scene_info-long.json")
    
    # run_MEM(task_level="easy", times = 5, arg = "--scenefile scene_info-tem.json")
    # run_MEM(task_level="basic", times = 5, arg = "--scenefile scene_info-long.json")
    
    
    # QWEN llm mem
    # run_LLM(task_level="basic", times = 5)
    # run_LLM(task_level="easy", times = 5)
    # run_MEM(task_level="basic", times = 5)
    run_MEM(task_level="easy", times = 5)

    pass
    