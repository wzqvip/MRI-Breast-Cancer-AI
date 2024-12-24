import requests
import json
import time
import os

#  config:
tcia_path = "../QIN-Breast/QIN-BREAST_2015-09-04.tcia"

#




def parse_tcia_file(filepath: str) -> dict:
    """
    解析 .tcia 文件，返回一个包含关键信息和系列UID列表的字典。
    """
    result = {}
    inside_series_list = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            # 如果行中含有"="，则解析为键值对
            if "=" in line:
                # 判断是否是 "ListOfSeriesToDownload=" 这一特殊键
                if line.startswith("ListOfSeriesToDownload="):
                    inside_series_list = True
                    result["ListOfSeriesToDownload"] = []
                    _, value = line.split("=", 1)
                    value = value.strip()
                    if value:
                        result["ListOfSeriesToDownload"].append(value)
                else:
                    # 普通键值对解析
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    result[key] = value
            else:
                # 如果已经进入Series列表，则这一行应当是SeriesInstanceUID
                if inside_series_list:
                    result["ListOfSeriesToDownload"].append(line)
    
    return result

def get_series_metadata(series_uid: str) -> dict:
    """
    使用 NBIA 的 getSeriesMetaData API 获取指定 SeriesInstanceUID 的详细信息。
    返回一个包含该 SeriesInstanceUID 元数据的字典。
    """
    base_url = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData'
    params = {'SeriesInstanceUID': series_uid}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        # API 返回列表，每个元素是一份元数据，我们一般只需第一条即可
        if isinstance(data, list):
            return data[0] if data else {}
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求 SeriesInstanceUID {series_uid} 失败: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"无法解析 SeriesInstanceUID {series_uid} 的响应内容。")
        return {}

def print_series_metadata(metadata: dict, series_number: int):
    """
    格式化并打印 Series 的详细信息，同时打印自定义的分类路径。
    """
    print(f"--- Series {series_number} ---")
    
    # 先把 NBIA 返回的元数据直接打印（可根据需要定制显示项）
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    # 获取要用于分类的关键字段
    collection         = metadata.get("Collection", "UnknownCollection")
    subject_id         = metadata.get("Subject ID", "UnknownSubjectID")  # NBIA元数据中多用 "PatientID" 表示
    series_description = metadata.get("Series Description", "NoDesc")
    study_uid          = metadata.get("Study UID", "UnknownStudyUID")
    study_date         = metadata.get("Study Date", "UnknownDate")
    
    # Study UID 的后五位
    last_5_digits = study_uid[-5:] if len(study_uid) >= 5 else study_uid
    # 将 series_description 拆分，保留第一个单词
    series_description = series_description.split()[0] if series_description else "NoDesc"

    # 构建分类字符串 (如 "04-11-1992-66691")
    # 注意：NBIA返回的StudyDate格式可能是 "MM-DD-YYYY" 或 "YYYY-MM-DD" 或者其他，需根据实际情况确认
    date_and_uid_suffix = f"{study_date}-{last_5_digits}"
    
    # 最终分类路径
    classification_path = os.path.join(
        collection,
        subject_id,
        date_and_uid_suffix,
        series_description
    )
    
    # 打印出分类路径信息
    print(f"分类路径: {classification_path}")
    print()

def main():

    
    # 解析 .tcia 文件
    parsed_data = parse_tcia_file(tcia_path)
    
    # 打印解析结果基本信息
    # print("解析结果：")
    # for key, val in parsed_data.items():
    #     if key == "ListOfSeriesToDownload":
    #         print(f"{key}:")
    #         for series_uid in val:
    #             print(f"  - {series_uid}")
    #     else:
            # print(f"{key} = {val}")
    
    # 获取并打印每个 SeriesInstanceUID 的详细信息
    series_uids = parsed_data.get("ListOfSeriesToDownload", [])
    
    if not series_uids:
        print("没有找到 SeriesInstanceUID 列表。")
        return
    
    print("\n获取每个 SeriesInstanceUID 的详细信息：\n")
    
    for idx, series_uid in enumerate(series_uids, start=1):
        metadata = get_series_metadata(series_uid)
        if metadata:
            print_series_metadata(metadata, idx)
        else:
            print(f"SeriesInstanceUID {series_uid} 没有返回有效的元数据。\n")
        
        # 为了避免请求过于频繁，可以适当暂停
        time.sleep(0.5)  # 暂停0.5秒

if __name__ == "__main__":
    main()
