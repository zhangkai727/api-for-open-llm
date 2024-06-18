import os
from pathlib import Path

import dotenv
import nltk

<<<<<<< HEAD
# 设置 NLTK_DATA_PATH 为当前脚本文件的父目录的父目录下的 'nltk_data' 文件夹路径
NLTK_DATA_PATH = os.path.join(Path(__file__).parents[1], "nltk_data")
# 将 NLTK_DATA_PATH 添加到 nltk.data.path 中，以确保 NLTK 能够找到数据
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# 加载 .env 文件，该文件位于当前脚本文件的父目录的父目录下
=======
NLTK_DATA_PATH = os.path.join(Path(__file__).parents[1], "nltk_data")
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
dotenv.load_dotenv(os.path.join(Path(__file__).parents[1], ".env"))
