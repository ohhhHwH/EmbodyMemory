import csv
import os
import pandas as pd

def evaluate_log(log_file_path)->dict:
    # 这里假设 log 文件是一个文本文件，包含多行，每行代表一个记录
    # 将文件名分割以提取任务名称等信息
    task_name = os.path.basename(log_file_path).replace(".log", "")
    parts = task_name.split("_")
    # 第一个为 难度
    # 第二个 - 倒数第四个 任务名称 （读到LLM 或 MEM 前为止）
    
    # 倒数第四个 LLM 或 MEM
    # 倒数第三个为 运行次数
    task_level = parts[0]
    task_type = parts[-4]
    task_actual_name = "_".join(parts[1:-4])
    task_times = parts[-3]

    Token_used = 131072 # 最大值
    check = False
    steps = 100
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Token used:" in line:
                Token_used = int(line.strip().split(":")[-1].strip())
            if "Check :" in line:
                success_str = line.strip().split(":")[-1].strip()
                check = success_str.lower() == "true"
            if "steps :" in line:
                steps = int(line.strip().split(":")[-1].strip())
    print(f"Evaluating Task: {task_actual_name}, Level: {task_level}, Type: {task_type}, Run: {task_times}")
    print(f"  Token Used: {Token_used}")
    print(f"  Success: {'Yes' if check else 'No'}")
    print(f"  Steps Taken: {steps}")
    print("-" * 40)
    
    return {
        "task_name": task_actual_name,
        "task_level": task_level,
        "task_type": task_type,
        "task_times": task_times,
        "token_used": Token_used,
        "success": check,
        "steps": steps
    }
    
def csv2xlsx(csv_file, xlsx_file):
    df = pd.read_csv(csv_file)
    df.to_excel(xlsx_file, index=False)  
    
def main():
    # 读取 文件夹 下文件
    folder_name = "./log/log-1222"
    
    # 遍历该文件夹下的所有文件
    eval_results = []
    for filename in os.listdir(folder_name):
        if filename.endswith(".log"):
            file_path = os.path.join(folder_name, filename)
            print(f"Evaluating log file: {file_path}")
            result = evaluate_log(file_path)
            eval_results.append(result)
    # 对eval_results进行排序
    eval_results.sort(key=lambda x: (x["task_level"], x["task_name"], int(x["task_times"]), x["task_type"]))
    
    # 遍历 eval_results 并统计出 dict {"task_name": success_count, avg_steps: x, avg_token: y}
    LLM_summary_results = {}
    MEM_summary_results = {}
    for res in eval_results:
        key = res["task_name"]
        if res["task_type"] == "LLM":
            summary_results = LLM_summary_results
        else:
            summary_results = MEM_summary_results
        if key not in summary_results:
            summary_results[key] = {
                "success_count": 0,
                "total_steps": 0,
                "total_token": 0,
                "total_runs": 0
            }
        summary_results[key]["total_runs"] += 1
        summary_results[key]["total_steps"] += res["steps"]
        summary_results[key]["total_token"] += res["token_used"]
        if res["success"]:
            summary_results[key]["success_count"] += 1
    
    # # 输出汇总结果 制作 xlsx 文件
    # # 将 LLM MEM 分别写入两个表
    llm_file = os.path.join(folder_name, "evaluation_LLM.csv")
    mem_file = os.path.join(folder_name, "evaluation_MEM.csv")
    with open(llm_file, 'w', newline='') as f_llm, open(mem_file, 'w', newline='') as f_mem:
        llm_writer = csv.writer(f_llm)
        mem_writer = csv.writer(f_mem)
        # 写入表头
        header = ["task_name", "task_level", "task_type", "task_times", "token_used", "success", "steps"]
        llm_writer.writerow(header)
        mem_writer.writerow(header)
        
        for res in eval_results:
            row = [res["task_name"], res["task_level"], res["task_type"], res["task_times"], res["token_used"], res["success"], res["steps"]]
            if res["task_type"] == "LLM":
                llm_writer.writerow(row)
            elif res["task_type"] == "MEM":
                mem_writer.writerow(row)
        header = ["task_name", "total_runs", "avg_token", "success_count", "avg_steps"]
        llm_writer.writerow(header)
        mem_writer.writerow(header)
        for key,value in LLM_summary_results.items():
            total_runs = LLM_summary_results[key]["total_runs"]
            avg_steps = LLM_summary_results[key]["total_steps"] / total_runs
            avg_token = LLM_summary_results[key]["total_token"] / total_runs
            success_count = LLM_summary_results[key]["success_count"]
            row = [key, total_runs, avg_token, success_count, avg_steps]
            llm_writer.writerow(row)
        for key,value in MEM_summary_results.items():
            total_runs = MEM_summary_results[key]["total_runs"]
            avg_steps = MEM_summary_results[key]["total_steps"] / total_runs
            avg_token = MEM_summary_results[key]["total_token"] / total_runs
            success_count = MEM_summary_results[key]["success_count"]
            row = [key, total_runs, avg_token, success_count, avg_steps]
            mem_writer.writerow(row)
            
    
    xlsx_llm_file = os.path.join(folder_name, "evaluation_LLM.xlsx")
    xlsx_mem_file = os.path.join(folder_name, "evaluation_MEM.xlsx")
    csv2xlsx(llm_file, xlsx_llm_file)
    csv2xlsx(mem_file, xlsx_mem_file)
    


if __name__ == "__main__":
    main()
