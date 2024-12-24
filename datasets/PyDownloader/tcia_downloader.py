import requests
import json
import os
import zipfile
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ======================
#       CONFIG
# ======================

# Path to the .tcia manifest file
TCIA_FILE_PATH = "../QIN-Breast/QIN-BREAST_2015-09-04.tcia"

# Maximum number of threads for parallel downloads
MAX_WORKERS = 1

# Whether to use the range-limited download
USE_RANGE = True
# If USE_RANGE is True, only download the items in [START_INDEX, END_INDEX)
START_INDEX = 0
END_INDEX = 10

# ======================
#   HELPER FUNCTIONS
# ======================

def parse_tcia_file(filepath: str) -> dict:
    """
    Parse a .tcia file to extract the configuration and the series UID list.
    """
    result = {}
    inside_series_list = False

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "=" in line:
                if line.startswith("ListOfSeriesToDownload="):
                    inside_series_list = True
                    result["ListOfSeriesToDownload"] = []
                    _, value = line.split("=", 1)
                    if value.strip():
                        result["ListOfSeriesToDownload"].append(value.strip())
                else:
                    key, value = line.split("=", 1)
                    result[key.strip()] = value.strip()
            else:
                if inside_series_list:
                    result["ListOfSeriesToDownload"].append(line)
    return result

def get_series_metadata(series_uid: str) -> dict:
    """
    Retrieve metadata for the given SeriesInstanceUID using NBIA API.
    """
    base_url = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData'
    params = {'SeriesInstanceUID': series_uid}
    
    try:
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return {}
    except Exception as e:
        print(f"[{series_uid}] ERROR: Failed to get metadata: {e}")
        return {}

def build_classification_path(metadata: dict) -> str:
    """
    Build a classification path based on metadata fields:
    Collection/SubjectID/StudyDate-StudyUIDLast5/SeriesDescFirstWord
    """
    collection = metadata.get("Collection", "UnknownCollection")
    subject_id = metadata.get("Subject ID", "UnknownSubjectID")
    study_uid  = metadata.get("Study UID", "UnknownStudyUID")
    study_date = metadata.get("Study Date", "UnknownDate")
    
    desc_full = metadata.get("Series Description", "NoDesc")
    desc_first = desc_full.split()[0] if desc_full else "NoDesc"
    
    uid_last_5 = study_uid[-5:] if len(study_uid) >= 5 else study_uid
    date_suffix = f"{study_date}-{uid_last_5}"
    
    path = os.path.join(collection, subject_id, date_suffix, desc_first)
    return path

def download_series_with_tqdm(series_uid: str, out_zip_path: str) -> bool:
    """
    Download the series as a ZIP file using requests + tqdm, and save to out_zip_path.
    Returns True if successful, else False.
    """
    base_download_url = 'https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage'
    download_url = f"{base_download_url}?SeriesInstanceUID={series_uid}"
    
    try:
        head_resp = requests.head(download_url)
        head_resp.raise_for_status()
        total_size = int(head_resp.headers.get('Content-Length', 0))
        
        with requests.get(download_url, stream=True) as r, open(out_zip_path, 'wb') as f:
            r.raise_for_status()
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True,
                                desc=f"Downloading {series_uid[-10:]}")

            chunk_size = 1024 * 256  # 256 KB
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
            progress_bar.close()
        return True
    except Exception as e:
        print(f"[{series_uid}] ERROR: Download failed: {e}")
        return False

def extract_zip_to_folder(zip_path: str, extract_dir: str) -> bool:
    """
    Extract the ZIP file to the target folder.
    Returns True if successful, else False.
    """
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path=extract_dir)
        return True
    except zipfile.BadZipFile as e:
        print(f"[{zip_path}] ERROR: Bad ZIP file: {e}")
    except Exception as e:
        print(f"[{zip_path}] ERROR: Extraction failed: {e}")
    return False

def download_and_extract_series(series_uid: str) -> None:
    """
    Complete process: get metadata -> build path -> download -> extract -> cleanup
    """
    metadata = get_series_metadata(series_uid)
    if not metadata:
        return
    
    classification_path = build_classification_path(metadata)
    os.makedirs(classification_path, exist_ok=True)
    
    zip_name = series_uid.replace('.', '_') + ".zip"
    zip_path = os.path.join(classification_path, zip_name)
    
    print(f"[{series_uid}] Download started -> {zip_path}")
    if not download_series_with_tqdm(series_uid, zip_path):
        return
    
    print(f"[{series_uid}] Extracting -> {classification_path}")
    if extract_zip_to_folder(zip_path, classification_path):
        print(f"[{series_uid}] Extraction done -> {classification_path}")
        # Optionally remove the ZIP if no longer needed
        try:
            os.remove(zip_path)
        except OSError:
            pass

# ======================
#        MAIN
# ======================
def main():
    parsed_data = parse_tcia_file(TCIA_FILE_PATH)
    series_uids = parsed_data.get("ListOfSeriesToDownload", [])
    if not series_uids:
        print("No Series UIDs found in the .tcia file.")
        return
    
    print(f"Found {len(series_uids)} Series UIDs.")
    
    # If range is enabled, filter the list
    if USE_RANGE:
        series_uids = series_uids[START_INDEX:END_INDEX]
        print(f"Using download range [{START_INDEX}, {END_INDEX}), total: {len(series_uids)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {executor.submit(download_and_extract_series, uid): uid for uid in series_uids}
        for future in as_completed(future_map):
            uid = future_map[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{uid}] ERROR: Unexpected exception: {e}")

if __name__ == "__main__":
    main()
