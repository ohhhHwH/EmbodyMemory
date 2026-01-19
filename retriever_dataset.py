
import os


# 对 memory/data/retriever 目录下的文件进行操作

def main():
    target_dir = "memory/data/retriever"
    retriever_dict = {}
    for filename in os.listdir(target_dir):
        # 提取文件第一行和第三行
        if filename.endswith(".txt"):
            file_path = os.path.join(target_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    first_line = lines[0].strip()
                    third_line = lines[2].strip()
                    print(f"File: {filename}")
                    print(f"  First line: {first_line}")
                    print(f"  Third line: {third_line}")
                    print("-" * 40)
                    # 将第一行 作为 key，第三行作为 value，存入字典
                    if first_line in retriever_dict:
                        # 如果 key 已存在，追加到 元组 中(不重复)
                        if third_line not in retriever_dict[first_line]:
                            retriever_dict[first_line] += (third_line,)
                    else:
                        retriever_dict[first_line] = (third_line,)
    # 将字典写入到 retriever_dataset.txt 文件中 于 memory/data 文件夹中
    output_file = "memory/data/retriever_dataset.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for key, values in retriever_dict.items():
            f.write(f"Key: {key}\n")
            for value in values:
                f.write(f"   {value}\n")

if __name__ == "__main__":
    main()
