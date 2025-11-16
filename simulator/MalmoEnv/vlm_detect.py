


import os
from openai import OpenAI
import requests
import base64
import requests
from openai import OpenAI
import base64
import json
import random
from PIL import Image, ImageDraw, ImageFont
import os
import base64
import time
from openai import OpenAI

prompt_qwen3_object_detection_en = """
you. Given an image, identify all objects present in the image and provide their details in a structured JSON format.
Please analyze the image and return a JSON array where each element contains the following fields for each detected.
object:
- "id": A unique identifier for the detected
- "label": The name of the detected object (e.g. "tree", "leaves").
- "confidence": A float value between 0 and 1 representing the confidence level of the detection.
- "bounding_box": An object containing the coordinates of the bounding box around the detected object with the following fields:
  - "x_min": The x-coordinate of the top-left corner of the bounding box.
  - "y_min": The y-coordinate of the top-left corner of the bounding box.
  - "x_max": The x-coordinate of the bottom-right corner of the bounding box.
  - "y_max": The y-coordinate of the bottom-right corner of the bounding box.
  # Ensure the JSON is properly formatted and can be easily parsed.
- "nearby:": A list of labels of objects that are in close proximity to the detected object.
Provide only the JSON array as the output without any additional text or explanation.
"""

prompt_qwen3_object_detection_cn = """
你是《我的世界》这个像素游戏中图像物体检测的专家。给定一张图片，识别图像中存在的物体（最多10个），并以结构化的JSON格式提供它们的详细信息。
请分析图片并返回一个JSON数组，其中每个元素包含以下字段，表示每个检测到的物体：
- "id"：检测到的物体的唯一标识符。
- "label"：检测到的物体的名称（例如：“树干”、“树叶”）。
- "bounding_box"：包含检测到的物体周围边界框坐标的对象，具有以下字段：
  - "x_min"：边界框左上角的x坐标。
  - "y_min"：边界框左上角的y坐标。
  - "x_max"：边界框右下角的x坐标。
  - "y_max"：边界框右下角的y坐标。
- "nearby:": 一个包含与检测到的物体接近的物体标签的列表。
仅提供JSON数组作为输出，不要添加任何额外的文本或解释。

"""
    
def test1():
    # 1. 首先将本地图片上传到临时存储或OSS，获取URL
    # 这里假设已经获取到图片URL
    image_url = "https://gitee.com/hou-yunlong817/imagehosting/blob/master/malmo_obs.png"

    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    def check_image_url(url):
        try:
            response = requests.head(url, timeout=10)
            print(f"URL: {url}")
            print(f"Status Code: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            print(f"Content-Length: {response.headers.get('Content-Length')}")
            return response.status_code == 200 and 'image' in response.headers.get('Content-Type', '')
        except Exception as e:
            print(f"Error checking URL: {e}")
            return False

    # 测试你的URL
    image_url = "https://gitee.com/hou-yunlong817/imagehosting/raw/master/malmo_depth.png"
    if check_image_url(image_url):
        print("✅ URL有效，可以继续调用API")
    else:
        print("❌ URL无效，请检查")

    response = client.chat.completions.create(
        model="qwen3-vl-plus",  # 使用Qwen3-VL-Plus模型
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片的内容"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )

    print(response.choices[0].message.content)
 
def test2():
    # 1. 读取本地图片文件并转换为base64编码
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # 假设你的图片在当前目录下的obs.png
    image_path = "malmo_obs.png"
    base64_image = image_to_base64(image_path)

    # 2. 构建Data URL格式
    # 根据图片类型选择正确的mime类型，PNG图片使用image/png
    data_url = f"data:image/png;base64,{base64_image}"  # [[8]]

    client = OpenAI(
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen3-vl-plus",  # 使用Qwen3-VL-Plus模型
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_qwen3_object_detection_cn},
                    {"type": "image_url", "image_url": {"url": data_url}}  # [[1]]
                ]
            }
        ]
    )
    print(response.choices[0].message.content)
    
    # 将 content 保存为 JSON 文件
    json_output_path = "detection_output.json"
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(response.choices[0].message.content)
    print(f"检测结果已保存到 {json_output_path}")

color_dir = {
    "tree": "#8B4513",
    "leaves": "#228B22",
    "water": "#1E90FF",
    "sand": "#C2B280",
    "grass": "#7CFC00",
    "dirt": "#654321",
    "stone": "#808080",
    "sky": "#87CEEB",
    "cloud": "#F0F8FF",
}

def draw_boxes_from_json():
    # 读取detection_output.json并绘制边框
    input_img = "malmo_rgb.png"       # 替换为你的图片路径
    output_img = "malmo_obs_annotated.png"
    json_path = "detection_output.json"   # 替换为你的JSON路径
    
    # 根据 img 实际尺寸设置参考尺寸（图片右下角坐标）
    with Image.open(input_img) as img:
        ref_width, ref_height = img.size
    
    # 绘制边框和加上文字
    
    font_size=36       # 字体大小（可调整）
    line_width=4       # 边框线宽
    # bbox_color="#FF0000"   # 边框颜色
    text_color="#FFFFFF"   # 文字颜色
    text_stroke_color="#000000"  # 文字描边颜色
    text_stroke_width=2      # 描边宽度
    
    # 读取 json 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)
    # 打开图片
    with Image.open(input_img) as img:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            print(f"Processing detection: {det}")
            box = det["bounding_box"]
            label = det["label"]
            x_min = int(box["x_min"])
            y_min = int(box["y_min"]) 
            x_max = int(box["x_max"])
            y_max = int(box["y_max"])
            
            # 绘制边框 - 随机边框颜色
            bbox_color = color_dir.get(label, "#FF0000")

            draw.rectangle([x_min, y_min, x_max, y_max], outline=bbox_color, width=line_width)
            
            # 绘制标签
            # label = f"{det['label']} ({det['confidence']:.2f})"
            # text_size = draw.textsize(label, font=font)
            # text_bg = [x_min, y_min - text_size[1], x_min + text_size[0], y_min]
            # draw.rectangle(text_bg, fill=bbox_color)
            # draw.text((x_min, y_min - text_size[1]), label, fill=text_color, font=font, stroke_width=text_stroke_width, stroke_fill=text_stroke_color)
        
        # 保存带有边框的图片
        img.save(output_img)
        print(f"已保存带有边框的图片到 {output_img}")
    
def draw_boxes_from_json_KIMI():
    # 读取detection_output_kimi.json并绘制边框
    input_img = "malmo_rgb.png"       # 替换为你的图片路径
    output_img = "malmo_obs_annotated.png"
    json_path = "detection_output_kimi.json"   # 替换为你的JSON路径
    
    # 根据 img 实际尺寸设置参考尺寸（图片右下角坐标）
    with Image.open(input_img) as img:
        ref_width, ref_height = img.size
    
    # 绘制边框和加上文字
    
    font_size=36       # 字体大小（可调整）
    line_width=4       # 边框线宽
    # bbox_color="#FF0000"   # 边框颜色
    text_color="#FFFFFF"   # 文字颜色
    text_stroke_color="#000000"  # 文字描边颜色
    text_stroke_width=2      # 描边宽度
    
    # 读取 json 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)
    # 打开图片
    with Image.open(input_img) as img:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for det in detections:
            print(f"Processing detection: {det}")
            box = det["bounding_box"]
            label = det["label"]
            if float(box["x_min"]) < 1 and float(box["x_max"]) < 1:
                x_min = float(box["x_min"]) * ref_width
                y_min = float(box["y_min"]) * ref_height
                x_max = float(box["x_max"]) * ref_width
                y_max = float(box["y_max"]) * ref_height
                # 将数据转换为整数
                det["bounding_box"]["x_min"] = x_min
                det["bounding_box"]["y_min"] = y_min
                det["bounding_box"]["x_max"] = x_max
                det["bounding_box"]["y_max"] = y_max
            else:
                x_min = int(box["x_min"])
                y_min = int(box["y_min"]) 
                x_max = int(box["x_max"])
                y_max = int(box["y_max"])
                
            # 绘制边框 - 随机边框颜色
            bbox_color = color_dir.get(label, "#FF0000")
            
            if x_min > x_max :
                x_min, x_max = x_max, x_min
            if y_min > y_max :
                y_min, y_max = y_max, y_min

            draw.rectangle([x_min, y_min, x_max, y_max], outline=bbox_color, width=line_width)
            
            # 绘制标签
            # label = f"{det['label']} ({det['confidence']:.2f})"
            # text_size = draw.textsize(label, font=font)
            # text_bg = [x_min, y_min - text_size[1], x_min + text_size[0], y_min]
            # draw.rectangle(text_bg, fill=bbox_color)
            # draw.text((x_min, y_min - text_size[1]), label, fill=text_color, font=font, stroke_width=text_stroke_width, stroke_fill=text_stroke_color)
        
        # 保存带有边框的图片
        img.save(output_img)
        print(f"已保存带有边框的图片到 {output_img}")
    
def test3_kimi():
    # kimi-latest-128k
    client = OpenAI(
        api_key = os.getenv("KIMI_API_KEY"), 
        base_url = "https://api.moonshot.cn/v1",
    )
    
    # 对图片进行base64编码
    image_path = "malmo_obs.png"
    with open(image_path, 'rb') as f:
        img_base = base64.b64encode(f.read()).decode('utf-8')
    
    start_time = time.time()
    response = client.chat.completions.create(
        model="moonshot-v1-8k-vision-preview", 
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_qwen3_object_detection_cn
                    }
                ]
            }
        ]
    )
    # print(response.choices[0].message.content)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # 检测 json 格式是否完整
    json_content = response.choices[0].message.content
    try:
        detections = json.loads(json_content)
        # print("JSON 格式正确")
    except json.JSONDecodeError as e:
        
        # 尝试修正 JSON 格式错误（根据具体错误进行调整）
        print("json_content:", json_content)
        
        print("JSON 格式错误:", e)
        
        # 删除 json 最后一个表项
        last_comma_index = json_content.rfind(',')
        if last_comma_index != -1:
            json_content = json_content[:last_comma_index] + json_content[last_comma_index + 1:]
        
    
    # 将 content 保存为 JSON 文件
    json_output_path = "detection_output_kimi.json"
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(response.choices[0].message.content)
        
    draw_boxes_from_json_KIMI()
    cal_pos()

# 根据框和深度图像计算物体位置
def cal_pos():
    # 深度图像路径
    depth_image_path = "malmo_depth.png"
    # 读取深度图像
    depth_image = Image.open(depth_image_path)
    depth_pixels = depth_image.load()
    
    # 遍历检测结果
    json_path = "detection_output_kimi.json"
    with open(json_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)
    for det in detections:
        box = det["bounding_box"]
        label = det["label"]
        if float(box["x_min"]) < 1 and float(box["x_max"]) < 1:
            x_min = int(float(box["x_min"]) * depth_image.width)
            y_min = int(float(box["y_min"]) * depth_image.height)
            x_max = int(float(box["x_max"]) * depth_image.width)
            y_max = int(float(box["y_max"]) * depth_image.height)
            
            # 将数据转换为整数
            det["bounding_box"]["x_min"] = x_min
            det["bounding_box"]["y_min"] = y_min
            det["bounding_box"]["x_max"] = x_max
            det["bounding_box"]["y_max"] = y_max
        else:
            x_min = int(box["x_min"])
            y_min = int(box["y_min"])
            x_max = int(box["x_max"])
            y_max = int(box["y_max"])
        
        # 计算边界框中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        # 获取深度值
        depth_value = depth_pixels[center_x, center_y]
        
        print(f"Object: {label}, Depth: {depth_value}")
        
        # json 中添加深度信息
        det["depth"] = depth_value
        det["x_mid"] = center_x
        det["y_mid"] = center_y
    
    # 将更新后的检测结果保存回 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(detections, f, ensure_ascii=False, indent=4)

def obj_detect_init():
    client = OpenAI(
        api_key = os.getenv("KIMI_API_KEY"), 
        base_url = "https://api.moonshot.cn/v1",
    )
    
    start_time = time.time()
    messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_qwen3_object_detection_cn
                    }
                ]
            }
        ]
    response = client.chat.completions.create(
        model="moonshot-v1-8k-vision-preview", 
        messages=messages
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds, response: {response}")
    return client, messages

def obj_detect_and_draw(client, messages, image_path="malmo_obs.png", model="moonshot-v1-8k-vision-preview"):

    start_time = time.time()
    
    messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')}"
                        }
                    }
                ]
            }
        )
    
    response = client.chat.completions.create(
        model=model, 
        messages=messages
    )
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
    # 将 content 保存为 JSON 文件
    json_output_path = "detection_output_kimi.json"
    with open(json_output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(response.choices[0].message.content)
        
    # 后续处理
    draw_boxes_from_json_KIMI()
    cal_pos()
    
    return messages


if __name__ == "__main__":
    cal_pos()