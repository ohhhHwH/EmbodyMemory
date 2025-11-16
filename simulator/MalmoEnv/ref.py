# 提示词：识别图中物体，给出json描述，格式为[{"catogory": "xxx", "description": "yyy", "bbox": [x1, y1, x2, y2]}, ...]。需包含物体坐标。
def draw_ref(
    input_image_path,
    output_image_path,
    json_path,
    ref_width,          # 参考尺寸宽度（图片右下角x坐标）
    ref_height,         # 参考尺寸高度（图片右下角y坐标）
    font_size=36,       # 字体大小（可调整）
    line_width=4,       # 边框线宽
    bbox_color="#FF0000",   # 边框颜色
    text_color="#FFFFFF",   # 文字颜色
    text_stroke_color="#000000",  # 文字描边颜色
    text_stroke_width=2      # 描边宽度
):
    # 读取JSON标注数据
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # 打开图片并获取实际尺寸
    with Image.open(input_image_path) as img:
        actual_width, actual_height = img.size
        draw = ImageDraw.Draw(img)
        scale_x = actual_width / ref_width
        scale_y = actual_height / ref_height
        
        # 重点：指定你的文泉驿正黑字体路径（已确认存在）
        font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        
        # 验证字体路径是否存在
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件不存在：{font_path}，请检查路径是否正确")
        
        # 加载中文字体
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"成功加载中文字体：{font_path}")
        except Exception as e:
            raise Exception(f"加载字体失败：{e}，可能是字体文件损坏或权限问题")
        
        # 遍历标注，绘制边框和居中文字
        for idx, ann in enumerate(annotations):
            category = ann.get("category", f"未知_{idx}")  # 中文类别
            ref_x1, ref_y1, ref_x2, ref_y2 = ann.get("bbox", [0,0,0,0])
            
            # 转换为实际坐标
            x1 = round(ref_x1 * scale_x)
            y1 = round(ref_y1 * scale_y)
            x2 = round(ref_x2 * scale_x)
            y2 = round(ref_y2 * scale_y)
            
            # 绘制边框
            draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=line_width)
            
            # 计算方框中心坐标
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            
            # 计算文字尺寸（用于居中对齐）
            text_bbox = draw.textbbox((0, 0), category, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 文字绘制起点（居中）
            text_x = bbox_center_x - text_width / 2
            text_y = bbox_center_y - text_height / 2
            
            # 绘制带描边的中文文字
            draw.text(
                (text_x, text_y),
                category,
                font=font,
                fill=text_color,
                stroke_width=text_stroke_width,
                stroke_fill=text_stroke_color
            )
        
        img.save(output_image_path)
    
    print(f"标注完成！共处理 {len(annotations)} 个条目，结果保存至：{output_image_path}")

# 使用示例
def ref():
    input_img = "1.jpg"       # 替换为你的图片路径
    output_img = "output.jpg"
    json_path = "a.json"   # 替换为你的JSON路径
    
    # 参考尺寸（图片右下角坐标）
    ref_width = 999    # 根据你的图片实际参考宽度设置
    ref_height = 999   # 根据你的图片实际参考高度设置
    
    # 调用函数（可调整字体大小）
    draw_ref(
        input_image_path=input_img,
        output_image_path=output_img,
        json_path=json_path,
        ref_width=ref_width,
        ref_height=ref_height,
        font_size=15  # 字体大小（根据方框大小调整）
    )
