

import os
from openai import OpenAI




# 将图像使用 qwen3 模型进行处理，使其返回json格式的物体检测结果


prompt_qwen3_object_detection = """
You are an expert in object detection from images. Given an image, identify all objects present in the image and provide their details in a structured JSON format.
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



def main():
    # 加载 qwen3 api
    client = OpenAI(
        api_key="your_api_key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    response = client.chat.completions.create(
        model="qwen3",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_qwen3_object_detection},
                    {"type": "image_url", "image_url": {"url": "https://example.com/your_image.jpg"}}
                ]
            }
        ]
    )
    print(response.choices[0].message.content)
    
    

if __name__ == "__main__":
    main()
