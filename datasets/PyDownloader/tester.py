import requests
import json
import zipfile
from io import BytesIO

def parse_tcia_file(filepath: str) -> dict:
    """
    解析 .tcia 文件，返回一个包含关键信息和系列UID列表的字典。
    
    返回字典示例：
    {
        'downloadServerUrl': 'https://public.cancerimagingarchive.net/nbia-download/servlet/DownloadServlet',
        'includeAnnotation': 'false',
        'noOfrRetry': '4',
        'databasketId': 'manifest-1542731172463.tcia',
        'manifestVersion': '3.0',
        'ListOfSeriesToDownload': [
            '1.3.6.1.4.1.14519.5.2.1.8162.7003.197953452011738747354955682782',
            '1.3.6.1.4.1.14519.5.2.1.8162.7003.272345977664722369820544794103',
            ...
        ]
    }
    """
    result = {}
    series_list = []
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
                    # 初始化 ListOfSeriesToDownload 列表
                    result["ListOfSeriesToDownload"] = []
                    # 检查等号后是否有内容（通常为空）
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
                # 如果不含有 "=" 并且前面检测到已经进入Series列表，则这一行应当是SeriesInstanceUID
                if inside_series_list:
                    result["ListOfSeriesToDownload"].append(line)
    
    return result

def get_series_metadata(series_uid: str) -> dict:
    """
    使用 NBIA 的 getSeriesMetaData API 获取指定 SeriesInstanceUID 的详细信息。
    
    返回示例：
    {
        "SeriesInstanceUID": "1.3.6.1.4.1.14519.5.2.1.8162.7003.197953452011738747354955682782",
        "Collection": "RIDER Lung CT",
        "PatientID": "RIDER-1129164940",
        "StudyInstanceUID": "1.3.6.1.4.1.9328.50.1.216116555221814778114703363464001196508",
        "Modality": "CT",
        ...
    }
    """
    base_url = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData'
    params = {
        'SeriesInstanceUID': series_uid
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        if isinstance(data, list):
            return data[0] if data else {}
        return data
    except requests.exceptions.RequestException as e:
        print(f"请求 SeriesInstanceUID {series_uid} 失败: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"无法解析 SeriesInstanceUID {series_uid} 的响应内容。")
        return {}

def main():
    # 假设 .tcia 文件的名称是 example.tcia
    tcia_path = "../QIN-Breast/QIN-BREAST_2015-09-04.tcia"
    
    # 解析 .tcia 文件
    parsed_data = parse_tcia_file(tcia_path)
    
    print("解析结果：")
    for key, val in parsed_data.items():
        if key == "ListOfSeriesToDownload":
            print(f"{key}:")
            for series_uid in val:
                print(f"  - {series_uid}")
        else:
            print(f"{key} = {val}")
    
    # 获取并打印每个 SeriesInstanceUID 的详细信息
    series_uids = parsed_data.get("ListOfSeriesToDownload", [])
    
    if not series_uids:
        print("没有找到 SeriesInstanceUID 列表。")
        return
    
    print("\n获取每个 SeriesInstanceUID 的详细信息：\n")
    
    for idx, series_uid in enumerate(series_uids, start=1):
        print(f"--- Series {idx} ---")
        metadata = get_series_metadata(series_uid)
        if metadata:
            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print(f"SeriesInstanceUID {series_uid} 没有返回有效的元数据。")
        print("\n")

if __name__ == "__main__":
    main()
