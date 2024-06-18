import importlib
from functools import lru_cache
from typing import (
    Dict,
    Any,
)

import chardet
import langchain_community.document_loaders
from langchain.text_splitter import (
    TextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.base import BaseLoader
from loguru import logger

from api.config import TEXT_SPLITTER_CONFIG

<<<<<<< HEAD
LOADER_MAPPINGS = {  # 定义一个字典，名为 LOADER_MAPPINGS，用于映射文件类型与对应的加载器
    "UnstructuredHTMLLoader":  # UnstructuredHTMLLoader 用于加载 HTML 文件
        [".html", ".htm"],  # 适用于 .html 和 .htm 文件
    "MHTMLLoader":  # MHTMLLoader 用于加载 MHTML 文件
        [".mhtml"],  # 适用于 .mhtml 文件
    "TextLoader":  # TextLoader 用于加载纯文本文件
        [".md"],  # 适用于 .md 文件
    "UnstructuredMarkdownLoader":  # UnstructuredMarkdownLoader 用于加载 Markdown 文件
        [".md"],
    "JSONLoader":  # JSONLoader 用于加载 JSON 文件
        [".json"],  # 适用于 .json 文件
    "JSONLinesLoader":  # JSONLinesLoader 用于加载 JSON 行文件
        [".jsonl"],
    "CSVLoader":  # CSVLoader 用于加载 CSV 文件
        [".csv"],  # 适用于 .csv 文件
    # "FilteredCSVLoader":  # 注释掉的 FilteredCSVLoader 用于加载带有自定义过滤条件的 CSV 文件
    "OpenParserPDFLoader":  # OpenParserPDFLoader 用于加载 PDF 文件
        [".pdf"],  # 适用于 .pdf 文件
    "RapidOCRPDFLoader":  # RapidOCRPDFLoader 用于通过 OCR 识别加载 PDF 文件
        [".pdf"],
    "RapidOCRDocLoader":  # RapidOCRDocLoader 用于通过 OCR 识别加载 Word 文档
        [".docx", ".doc"],  # 适用于 .docx 和 .doc 文件
    "RapidOCRPPTLoader":  # RapidOCRPPTLoader 用于通过 OCR 识别加载 PowerPoint 文档
        [".ppt", ".pptx"],  # 适用于 .ppt 和 .pptx 文件
    "RapidOCRLoader":  # RapidOCRLoader 用于通过 OCR 识别加载图像文件
        [".png", ".jpg", ".jpeg", ".bmp"],  # 适用于 .png, .jpg, .jpeg, .bmp 文件
    "UnstructuredFileLoader":  # UnstructuredFileLoader 用于加载多种类型的未结构化文件
        [".eml", ".msg", ".rst", ".rtf", ".txt", ".xml", ".epub", ".odt", ".tsv"],  # 适用于多种文件类型
    "UnstructuredEmailLoader":  # UnstructuredEmailLoader 用于加载电子邮件文件
        [".eml", ".msg"],  # 适用于 .eml 和 .msg 文件
    "UnstructuredEPubLoader":  # UnstructuredEPubLoader 用于加载电子书文件
        [".epub"],  # 适用于 .epub 文件
    "UnstructuredExcelLoader":  # UnstructuredExcelLoader 用于加载 Excel 文件
        [".xlsx", ".xls", ".xlsd"],  # 适用于 .xlsx, .xls, .xlsd 文件
    "NotebookLoader":  # NotebookLoader 用于加载 Jupyter Notebook 文件
        [".ipynb"],  # 适用于 .ipynb 文件
    "UnstructuredODTLoader":  # UnstructuredODTLoader 用于加载 OpenDocument 文本文件
        [".odt"],  # 适用于 .odt 文件
    "PythonLoader":  # PythonLoader 用于加载 Python 脚本文件
        [".py"],  # 适用于 .py 文件
    "UnstructuredRSTLoader":  # UnstructuredRSTLoader 用于加载 reStructuredText 文件
        [".rst"],  # 适用于 .rst 文件
    "UnstructuredRTFLoader":  # UnstructuredRTFLoader 用于加载 RTF 文件
        [".rtf"],  # 适用于 .rtf 文件
    "SRTLoader":  # SRTLoader 用于加载 SRT 字幕文件
        [".srt"],  # 适用于 .srt 文件
    "TomlLoader":  # TomlLoader 用于加载 TOML 文件
        [".toml"],  # 适用于 .toml 文件
    "UnstructuredTSVLoader":  # UnstructuredTSVLoader 用于加载 TSV 文件
        [".tsv"],  # 适用于 .tsv 文件
    "UnstructuredWordDocumentLoader":  # UnstructuredWordDocumentLoader 用于加载 Word 文档
        [".docx", ".doc"],  # 适用于 .docx 和 .doc 文件
    "UnstructuredXMLLoader":  # UnstructuredXMLLoader 用于加载 XML 文件
        [".xml"],  # 适用于 .xml 文件
    "UnstructuredPowerPointLoader":  # UnstructuredPowerPointLoader 用于加载 PowerPoint 文档
        [".ppt", ".pptx"],  # 适用于 .ppt 和 .pptx 文件
    "EverNoteLoader":  # EverNoteLoader 用于加载 EverNote 文件
        [".enex"],  # 适用于 .enex 文件
}


SUPPORTED_EXTS = [  # 定义一个列表，名为 SUPPORTED_EXTS，用于存储所有支持的文件扩展名
    ext for sublist in LOADER_MAPPINGS.values() for ext in sublist  # 使用列表解析，从 LOADER_MAPPINGS 字典中提取所有文件扩展名
]


class JSONLinesLoader(JSONLoader):  # 定义一个名为 JSONLinesLoader 的类，继承自 JSONLoader 类
    def __init__(self, *args, **kwargs):  # 定义类的构造函数，接受任意数量的位置参数和关键字参数
        super().__init__(*args, **kwargs)  # 调用父类 JSONLoader 的构造函数，传递所有参数
        self._json_lines = True  # 设置实例属性 _json_lines 为 True

langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader  # 将 JSONLinesLoader 类赋值给 langchain_community.document_loaders 模块中的 JSONLinesLoader


def get_loader_class(file_extension):  # 定义一个名为 get_loader_class 的函数，接受一个参数 file_extension
    for cls, exts in LOADER_MAPPINGS.items():  # 遍历 LOADER_MAPPINGS 字典的每一项，cls 是键（类名），exts 是值（扩展名列表）
        if file_extension in exts:  # 检查 file_extension 是否在当前的扩展名列表 exts 中
            return cls  # 如果找到了匹配的扩展名，返回对应的类名 cls


def get_loader(  # 定义一个名为 get_loader 的函数
    loader_name: str,  # 接受参数 loader_name，类型为字符串
    file_path: str,  # 接受参数 file_path，类型为字符串
    loader_kwargs: Dict[str, Any] = None,  # 接受参数 loader_kwargs，类型为字典，默认为 None
) -> BaseLoader:  # 返回类型为 BaseLoader
    """ 根据 loader_name 和文件路径或内容返回文档加载器 """
    loader_kwargs = loader_kwargs or {}  # 如果 loader_kwargs 为空，则初始化为一个空字典
    try:
        if loader_name in [  # 如果 loader_name 在以下列表中
=======
LOADER_MAPPINGS = {
    "UnstructuredHTMLLoader":
        [".html", ".htm"],
    "MHTMLLoader":
        [".mhtml"],
    "TextLoader":
        [".md"],
    "UnstructuredMarkdownLoader":
        [".md"],
    "JSONLoader":
        [".json"],
    "JSONLinesLoader":
        [".jsonl"],
    "CSVLoader":
        [".csv"],
    # "FilteredCSVLoader":
    #     [".csv"], 如果使用自定义分割csv
    "OpenParserPDFLoader":
        [".pdf"],
    "RapidOCRPDFLoader":
        [".pdf"],
    "RapidOCRDocLoader":
        [".docx", ".doc"],
    "RapidOCRPPTLoader":
        [".ppt", ".pptx", ],
    "RapidOCRLoader":
        [".png", ".jpg", ".jpeg", ".bmp"],
    "UnstructuredFileLoader":
        [".eml", ".msg", ".rst", ".rtf", ".txt", ".xml", ".epub", ".odt", ".tsv"],
    "UnstructuredEmailLoader":
        [".eml", ".msg"],
    "UnstructuredEPubLoader":
        [".epub"],
    "UnstructuredExcelLoader":
        [".xlsx", ".xls", ".xlsd"],
    "NotebookLoader":
        [".ipynb"],
    "UnstructuredODTLoader":
        [".odt"],
    "PythonLoader":
        [".py"],
    "UnstructuredRSTLoader":
        [".rst"],
    "UnstructuredRTFLoader":
        [".rtf"],
    "SRTLoader":
        [".srt"],
    "TomlLoader":
        [".toml"],
    "UnstructuredTSVLoader":
        [".tsv"],
    "UnstructuredWordDocumentLoader":
        [".docx", ".doc"],
    "UnstructuredXMLLoader":
        [".xml"],
    "UnstructuredPowerPointLoader":
        [".ppt", ".pptx"],
    "EverNoteLoader":
        [".enex"],
}

SUPPORTED_EXTS = [
    ext for sublist in LOADER_MAPPINGS.values() for ext in sublist
]


class JSONLinesLoader(JSONLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_loader_class(file_extension):
    for cls, exts in LOADER_MAPPINGS.items():
        if file_extension in exts:
            return cls


def get_loader(
    loader_name: str,
    file_path: str,
    loader_kwargs: Dict[str, Any] = None,
) -> BaseLoader:
    """ 根据 loader_name 和文件路径或内容返回文档加载器 """
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in [
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
            "OpenParserPDFLoader",
            "RapidOCRPDFLoader",
            "RapidOCRLoader",
            "FilteredCSVLoader",
            "RapidOCRDocLoader",
            "RapidOCRPPTLoader",
        ]:
<<<<<<< HEAD
            loaders_module = importlib.import_module(  # 导入 api.rag.processors.loader 模块
                "api.rag.processors.loader"
            )
        else:  # 否则
            loaders_module = importlib.import_module(  # 导入 langchain_community.document_loaders 模块
                "langchain_community.document_loaders"
            )
        DocumentLoader = getattr(loaders_module, loader_name)  # 从模块中获取名为 loader_name 的类

    except Exception as e:  # 捕捉异常
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"  # 构建错误消息
        logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)  # 记录错误日志
        loaders_module = importlib.import_module(  # 导入 langchain_community.document_loaders 模块
            "langchain_community.document_loaders"
        )
        DocumentLoader = getattr(loaders_module, "UnstructuredFileLoader")  # 获取 UnstructuredFileLoader 类

    if loader_name == "UnstructuredFileLoader":  # 如果 loader_name 是 UnstructuredFileLoader
        loader_kwargs.setdefault("autodetect_encoding", True)  # 设置默认参数 autodetect_encoding 为 True

    elif loader_name == "CSVLoader":  # 如果 loader_name 是 CSVLoader
        if not loader_kwargs.get("encoding"):  # 如果未指定 encoding 参数
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:  # 打开文件
                encode_detect = chardet.detect(struct_file.read())  # 检测文件编码
            if encode_detect is None:  # 如果未检测到编码
                encode_detect = {"encoding": "utf-8"}  # 设置默认编码为 utf-8
            loader_kwargs["encoding"] = encode_detect["encoding"]  # 设置 encoding 参数

    elif loader_name == "JSONLoader":  # 如果 loader_name 是 JSONLoader
        loader_kwargs.setdefault("jq_schema", ".")  # 设置默认参数 jq_schema 为 "."
        loader_kwargs.setdefault("text_content", False)  # 设置默认参数 text_content 为 False

    elif loader_name == "JSONLinesLoader":  # 如果 loader_name 是 JSONLinesLoader
        loader_kwargs.setdefault("jq_schema", ".")  # 设置默认参数 jq_schema 为 "."
        loader_kwargs.setdefault("text_content", False)  # 设置默认参数 text_content 为 False

    loader = DocumentLoader(file_path, **loader_kwargs)  # 创建 DocumentLoader 实例
    return loader  # 返回 DocumentLoader 实例
=======
            loaders_module = importlib.import_module(
                "api.rag.processors.loader"
            )
        else:
            loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
        DocumentLoader = getattr(loaders_module, loader_name)

    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}", exc_info=e)
        loaders_module = importlib.import_module(
            "langchain_community.document_loaders"
        )
        DocumentLoader = getattr(loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)

    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d


@lru_cache()
def make_text_splitter(
    splitter_name: str, chunk_size: int, chunk_overlap: int
) -> TextSplitter:
    """ 根据参数获取特定的分词器 """
<<<<<<< HEAD
    splitter_name = splitter_name or "SpacyTextSplitter"  # 如果没有提供分词器名称，使用默认值 "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # 如果分词器名称是 "MarkdownHeaderTextSplitter"，进行特殊处理
            headers_to_split_on = TEXT_SPLITTER_CONFIG[splitter_name]["headers_to_split_on"]  # 获取配置中的分割头信息
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False  # 创建 MarkdownHeaderTextSplitter 实例
            )
        else:
            try:  # 尝试导入用户自定义的分词器模块
                text_splitter_module = importlib.import_module(
                    "api.rag.processors.splitter"
                )
            except ImportError:  # 如果导入失败，则导入 langchain 的分词器模块
=======
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = TEXT_SPLITTER_CONFIG[splitter_name]["headers_to_split_on"]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
        else:
            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module(
                    "api.rag.processors.splitter"
                )
            except ImportError:  # 否则使用langchain的text_splitter
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )

<<<<<<< HEAD
            TextSplitter = getattr(text_splitter_module, splitter_name)  # 获取分词器类

            if TEXT_SPLITTER_CONFIG[splitter_name]["source"] == "tiktoken":  # 如果分词器来源是 tiktoken
=======
            TextSplitter = getattr(text_splitter_module, splitter_name)

            if TEXT_SPLITTER_CONFIG[splitter_name]["source"] == "tiktoken":  # 从tiktoken加载
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

<<<<<<< HEAD
            elif TEXT_SPLITTER_CONFIG[splitter_name]["source"] == "huggingface":  # 如果分词器来源是 huggingface
                if TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")  # 使用 GPT2 的分词器
                else:  # 使用字符长度加载分词器
=======
            elif TEXT_SPLITTER_CONFIG[splitter_name]["source"] == "huggingface":  # 从huggingface加载
                if TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  # 字符长度加载
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        TEXT_SPLITTER_CONFIG[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
    except Exception as e:
<<<<<<< HEAD
        logger.error(e)  # 如果在创建分词器过程中出现异常，记录错误日志
=======
        logger.error(e)
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d
        text_splitter_module = importlib.import_module(
            "langchain.text_splitter"
        )
        TextSplitter = getattr(
            text_splitter_module, "RecursiveCharacterTextSplitter"
        )
        text_splitter = TextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

<<<<<<< HEAD
    return text_splitter  # 返回创建的分词器实例

=======
    return text_splitter
>>>>>>> d45db7c71cc1d7c6f454aab8dc32da6b0299ee3d


