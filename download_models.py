import gdown
import os

# The Google Drive Folder ID provided by the user
folder_id = '1mMI11ZXiH_TvJUMigcZbLyeEsYoBV8c_'

print("Starting automatic model download from Google Drive...")
# This will download the folder contents
gdown.download_folder(id=folder_id, quiet=False, use_cookies=False)
print("Download completed successfully!")
