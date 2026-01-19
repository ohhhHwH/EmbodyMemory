


import os

folder_name = "./log/log-0101"
temp_folder = "./log/log-0101/tem"

# 遍历 folder_name 下的所有 easy_craft_wooden_slab_plate_MEM_1_251231_2119.log 文件 根据后缀时间选出最早的50个运行的文件放入 ./log/log-0101/tem 文件夹下

def main():
    files = os.listdir(folder_name)
    log_files = [f for f in files if f.endswith(".log")]
    
    # 提取时间戳并排序
    def extract_timestamp(filename):
        parts = filename.split('_')
        if len(parts) < 6:
            return ""
        return parts[-3] + parts[-2]  # mmyydd_hhmm
    
    log_files.sort(key=extract_timestamp)
    
    # 选择最早的50个文件
    selected_files = log_files[:50]
    
    # 创建临时文件夹
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    # 移动文件到临时文件夹
    for file in selected_files:
        src_path = os.path.join(folder_name, file)
        dst_path = os.path.join(temp_folder, file)
        os.rename(src_path, dst_path)
        print(f"Moved: {file} to {temp_folder}")
    
    

if __name__ == "__main__":
    main()