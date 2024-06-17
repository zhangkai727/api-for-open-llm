import os
import secrets
from typing import (
    List,
    Optional,
    Any,
)

import requests
from fastapi import APIRouter, HTTPException, UploadFile
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from openai.pagination import SyncPage
from openai.types.file_deleted import FileDeleted
from openai.types.file_object import FileObject
from pydantic import BaseModel

from api.config import STORAGE_LOCAL_PATH
from api.rag.processors import (
    get_loader,
    make_text_splitter,
    get_loader_class,
)
from api.rag.processors.splitter import zh_title_enhance as func_zh_title_enhance

file_router = APIRouter(prefix="/files")


class File2DocsRequest(BaseModel):
    file_id: Optional[str] = None
    # 文件ID，初始值为None

    url: Optional[str] = None
    # URL，初始值为None

    zh_title_enhance: Optional[bool] = False
    # 是否增强中文标题，默认为False

    chunk_size: Optional[int] = 250
    # 用于文本分块，默认为250

    chunk_overlap: Optional[int] = 50

    text_splitter_name: Optional[str] = "ChineseRecursiveTextSplitter"
    # 文本分割器名称，默认为"ChineseRecursiveTextSplitter"

    url_parser_prefix: Optional[str] = "https://r.jina.ai/"
    # URL解析器前缀，默认为"https://r.jina.ai/"



class File2DocsResponse(BaseModel):
    id: str
    # 文档响应的唯一标识符，类型为字符串

    object: str = "docs"
    #对象类型，默认为"docs"

    docs: List[Any]
    # 文档列表，类型为任意类型的列表



@file_router.post("", response_model=FileObject)
async def upload_file(file: UploadFile):
    # 生成文件ID，格式为"file-" + 12位随机十六进制字符串（用下划线替换短横线）
    file_id = "file-" + str(secrets.token_hex(12)).replace("-", "_")

    # 获取上传文件的原始文件名
    filename = file.filename

    # 构建文件在本地存储的完整路径
    filepath = os.path.join(STORAGE_LOCAL_PATH, f"{file_id}_{filename}")

    # 将上传的文件内容写入到本地文件中
    with open(filepath, "wb") as f:
        f.write(file.file.read())

    # 构建并返回FileObject对象作为响应
    return FileObject(
        id=file_id,  # 文件ID
        bytes=os.path.getsize(filepath),  # 文件大小（字节数）
        created_at=int(os.path.getctime(filepath)),  # 文件创建时间（Unix时间戳）
        filename=filename,  # 文件名
        object="file",  # 对象类型（文件）
        purpose="assistants",  # 文件用途
        status="uploaded",  # 文件状态（已上传）
    )



@file_router.get("/{file_id}", response_model=FileObject)
async def get_details(file_id: str):
    # 调用内部函数 _find_file 查找具有给定文件ID的文件
    file = _find_file(file_id)

    # 如果找到了文件
    if file:
        # 解析文件ID和文件名
        file_id = file.split("_")[0]
        filename = "_".join(file.split("_")[1:])

        # 构建文件在本地存储的完整路径
        filepath = os.path.join(STORAGE_LOCAL_PATH, file)

        # 构建并返回FileObject对象作为响应
        return FileObject(
            id=file_id,  # 文件ID
            bytes=os.path.getsize(filepath),  # 文件大小（字节数）
            created_at=int(os.path.getctime(filepath)),  # 文件创建时间（Unix时间戳）
            filename=filename,  # 文件名
            object="file",  # 对象类型（文件）
            purpose="assistants",  # 文件用途
            status="uploaded",  # 文件状态（已上传）
        )
    else:
        # 如果未找到文件，则抛出HTTP 404异常
        raise HTTPException(status_code=404, detail=f"File {file_id} not found!")


@file_router.get("")
async def list_files():
    data = []  # 初始化一个空列表，用于存储文件对象信息
    # 遍历存储路径下的所有文件
    for file in os.listdir(STORAGE_LOCAL_PATH):
        # 解析文件ID和文件名
        file_id = file.split("_")[0]
        filename = "_".join(file.split("_")[1:])

        # 构建文件在本地存储的完整路径
        filepath = os.path.join(STORAGE_LOCAL_PATH, file)

        # 构建一个新的FileObject对象，并添加到data列表中
        data.append(
            FileObject(
                id=file_id,  # 文件ID
                bytes=os.path.getsize(filepath),  # 文件大小（字节数）
                created_at=int(os.path.getctime(filepath)),  # 文件创建时间（Unix时间戳）
                filename=filename,  # 文件名
                object="file",  # 对象类型（文件）
                purpose="assistants",  # 文件用途
                status="uploaded",  # 文件状态（已上传）
            )
        )

    # 返回一个SyncPage对象，表示同步页的数据列表
    return SyncPage(data=data, object="list")


@file_router.delete("/{file_id}", response_model=FileDeleted)
async def delete_file(file_id: str):

    deleted = False  # 初始化删除状态为False

    # 查找具有给定文件ID的文件名
    filename = _find_file(file_id)

    # 如果找到了文件名
    if filename:
        # 构建文件在本地存储的完整路径
        filepath = os.path.join(STORAGE_LOCAL_PATH, filename)

        # 如果文件路径存在
        if filepath:
            # 删除文件
            os.remove(filepath)
            deleted = True  # 标记文件删除成功

        # 返回FileDeleted对象作为响应
        return FileDeleted(id=file_id, object="file", deleted=deleted)
    else:
        # 如果未找到文件，则抛出HTTP 404异常
        raise HTTPException(status_code=404, detail=f"File {file_id} not found!")


@file_router.post("/split", response_model=File2DocsResponse)
async def split_into_docs(request: File2DocsRequest):
    """
    处理将文件拆分为文档的POST请求，并返回File2DocsResponse对象作为响应模型。
    """

    if request.url is not None:
        # 如果请求中包含URL
        try:
            headers = {"Accept": "application/json"}
            # 发送GET请求获取URL内容
            res = requests.get(f"{request.url_parser_prefix}{request.url}", headers=headers).json()

            # 从响应中提取文档内容
            docs = [
                Document(page_content=res["data"]["content"])
            ]

            # 设置扩展名为空字符串
            ext = ""
            # 设置来源信息，包括URL和标题（如果存在）
            source = {"url": request.url, "title": res["data"].get("title")}

        except:
            # 如果解析URL失败，则抛出HTTP 404异常
            raise HTTPException(status_code=404, detail=f"Parsing {request.url} failed!")

    else:
        # 如果请求中没有URL，则处理文件ID
        filename = _find_file(request.file_id)

        # 如果未找到文件，则抛出HTTP 404异常
        if not filename:
            raise HTTPException(status_code=404, detail=f"File {request.file_id} not found!")

        # 构建文件在本地存储的完整路径
        filepath = os.path.join(STORAGE_LOCAL_PATH, filename)
        # 获取文件的扩展名，并转换为小写
        ext = os.path.splitext(filepath)[-1].lower()

        # 根据文件扩展名获取相应的加载器
        loader = get_loader(loader_name=get_loader_class(ext), file_path=filepath)

        # 如果是文本加载器
        if isinstance(loader, TextLoader):
            loader.encoding = "utf8"
            # 加载文档内容
            docs = loader.load()
        else:
            # 加载文档内容
            docs = loader.load()

        # 设置来源信息，包括文件ID和文件名
        source = {
            "file_id": request.file_id,
            "filename": "_".join(filename.split("_")[1:])
        }

    # 如果未加载到文档，则返回空列表
    if not docs:
        return []

    # 如果文件扩展名不在支持的列表中（例如不是.csv文件）
    if ext not in [".csv"]:
        # 根据请求中的文本分割器名称创建文本分割器
        text_splitter = make_text_splitter(
            splitter_name=request.text_splitter_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )

        # 如果使用的是MarkdownHeaderTextSplitter文本分割器
        if request.text_splitter_name == "MarkdownHeaderTextSplitter":
            # 对单个文档内容进行文本分割
            docs = text_splitter.split_text(docs[0].page_content)
        else:
            # 对多个文档进行文本分割
            docs = text_splitter.split_documents(docs)

    # 如果未加载到文档，则返回空列表
    if not docs:
        return []

    # 如果请求中指定了增强中文标题
    if request.zh_title_enhance:
        # 对文档内容进行中文标题增强处理
        docs = func_zh_title_enhance(docs)

    # 构建并返回File2DocsResponse对象作为响应
    return File2DocsResponse(
        id="docs-" + str(secrets.token_hex(12)),  # 分配一个唯一的ID作为文件组ID
        docs=[
            dict(
                page_content=d.page_content,  # 文档内容
                metadata={"source": source},  # 元数据，包括来源信息
                type="Document",  # 文档类型
            )
            for d in docs
        ]
    )


def _find_file(file_id: str):

    files = os.listdir(STORAGE_LOCAL_PATH)  # 获取存储路径下的所有文件列表
    for file in files:
        if file.startswith(file_id):  # 判断文件名是否以给定的文件ID开头
            return file  # 返回找到的文件名
    return None  # 如果未找到匹配的文件名，则返回None

