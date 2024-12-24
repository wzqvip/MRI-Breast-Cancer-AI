# MRI-Breast-Cancer-AI

 A comprehensive project leveraging deep learning and advanced MRI image analysis techniques for breast cancer detection, segmentation, and prognosis prediction.



## TCIA Downloader

The official TCIA downloader only works in single-thread mode and lacks speed visibility. 

So we made our own Python based downloader.

***tcia_downloader** *addresses these issues by:

* **Multi-Threaded Downloads** : Accelerates large-scale downloads using Python’s `ThreadPoolExecutor`.
* **Progress Bar** : Shows real-time download progress via `tqdm`.
* **Configurable Structure** : Organizes files into a user-defined folder hierarchy (e.g., Collection/Subject/StudyUID/…).
* **Optional Range** : Allows downloading specific segments of the `.tcia` manifest if needed.
* **Simple Setup** : Key settings (file path, thread count, etc.) are defined at the top of the script for easy editing.

Use with care! DO NOT VIOLATE the rules from the data provider!
