"""
WSI 标注解析模块 (WSI Annotation Preprocessor)
用于解析病理全切片图像 (WSI) 的 XML 标注文件，提取多边形区域、坐标顶点以及分类标签，
并将其结构化为下游切片生成 (Slice Generation) 任务所需的格式。
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional


def parse_pathology_xml(xml_path: str) -> Dict:
    """
    解析病理切片 XML 标注文件，返回结构化信息字典。
    
    Args:
        xml_path (str): WSI 对应的 XML 标注文件路径。
        
    Returns:
        Dict: 包含分辨率信息和所有标注区域（坐标、面积、标签等）的结构化字典。
        
    Raises:
        FileNotFoundError: 如果指定的 XML 文件不存在。
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML 文件未找到: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 1. 提取 MicronsPerPixel (微米/像素比例)
    microns_per_pixel = float(root.attrib.get("MicronsPerPixel", 0))

    results = {
        "MicronsPerPixel": microns_per_pixel,
        "Annotations": []
    }

    # 2. 遍历每个 Annotation
    for annotation in root.findall(".//Annotation"):
        ann_id = annotation.attrib.get("Id")
        ann_name = annotation.attrib.get("Name", "")

        # 遍历 Regions
        for region in annotation.findall(".//Region"):
            region_data = {
                "RegionID": region.attrib.get("Id"),
                "Type": region.attrib.get("Type"),  # 0=多边形
                "Zoom": float(region.attrib.get("Zoom", 1.0)),
                "LengthPixels": float(region.attrib.get("Length", 0)),
                "AreaPixels": float(region.attrib.get("Area", 0)),
                "LengthMicrons": float(region.attrib.get("LengthMicrons", 0)),
                "AreaMicrons": float(region.attrib.get("AreaMicrons", 0)),
                "Text": region.attrib.get("Text", ""),
                "NegativeROA": int(region.attrib.get("NegativeROA", 0)),
                "Description": "",
                "Vertices": []  # 格式: List[Tuple[float, float]]
            }

            # 提取 Description（包含病理分类的自定义属性）
            for attr in region.findall(".//Attribute"):
                if attr.attrib.get("Name") == "Description" or attr.attrib.get("Id") == "1":
                    region_data["Description"] = attr.attrib.get("Value", "")
                    break

            # 提取多边形的所有顶点坐标
            for vertex in region.findall(".//Vertex"):
                x = float(vertex.attrib.get("X"))
                y = float(vertex.attrib.get("Y"))
                region_data["Vertices"].append((x, y))

            results["Annotations"].append({
                "AnnotationID": ann_id,
                "AnnotationName": ann_name,
                "Region": region_data
            })

    return results


def print_xml_summary(data: Dict) -> None:
    """
    打印解析结果的中文摘要，用于数据探查和日志记录。
    
    Args:
        data (Dict): 由 parse_pathology_xml 解析得到的结构化字典。
    """
    mpp = data.get("MicronsPerPixel", 0)
    print(f"图像分辨率: {mpp} 微米/像素")
    print(f"共检测到 {len(data['Annotations'])} 个标注区域\n")
    print("=" * 60)

    for i, ann in enumerate(data["Annotations"]):
        region = ann["Region"]
        desc = region["Description"] or "无标签"
        area_um = region["AreaMicrons"]
        length_um = region["LengthMicrons"]
        vertex_count = len(region["Vertices"])

        print(f"【标注 {i + 1}】")
        print(f"  Annotation ID: {ann['AnnotationID']}")
        print(f"  区域 ID: {region['RegionID']}")
        print(f"  标签: {desc}")
        print(f"  面积: {area_um:.1f} μm²")
        print(f"  周长: {length_um:.1f} μm")
        print(f"  顶点数: {vertex_count}")
        if vertex_count > 0:
            print(f"  首尾坐标: ({region['Vertices'][0][0]:.0f}, {region['Vertices'][0][1]:.0f}) → "
                  f"({region['Vertices'][-1][0]:.0f}, {region['Vertices'][-1][1]:.0f})")
        print("-" * 40)


def save_to_json(data: Dict, output_path: str) -> None:
    """
    将解析后的结构化数据导出为 JSON 文件，供后续流程读取。
    
    Args:
        data (Dict): 结构化数据字典。
        output_path (str): 目标 JSON 文件的保存路径。
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 解析结果已成功导出至: {output_path}")


if __name__ == "__main__":
    # 使用 argparse 增加命令行支持，便于集成到工程 pipeline 中
    parser = argparse.ArgumentParser(description="WSI XML 标注文件解析工具")
    parser.add_argument(
        "--xml_path", 
        type=str, 
        default="path/to/your/dataset/annotations/sample.xml", # 外部数据集占位符
        help="输入的 XML 标注文件路径"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="path/to/your/dataset/processed/annotation_output.json", # 外部输出占位符
        help="解析结果的 JSON 导出路径"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true", 
        help="是否关闭标准输出摘要打印"
    )

    args = parser.parse_args()

    try:
        # 1. 解析 XML
        parsed_data = parse_pathology_xml(args.xml_path)
        
        # 2. 打印摘要 (除非使用 --quiet)
        if not args.quiet:
            print_xml_summary(parsed_data)

        # 3. 导出 JSON
        save_to_json(parsed_data, args.output_path)

    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
        print("💡 提示: 请确保传入了真实存在的 XML 文件路径。例如: python preprocess.py --xml_path data/raw/A10.xml")
    except Exception as e:
        print(f"❌ 解析过程中发生未知错误: {e}")