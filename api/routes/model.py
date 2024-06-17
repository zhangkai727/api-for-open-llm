import time
from typing import List

from fastapi import APIRouter, Depends, status
from openai.types.model import Model
from pydantic import BaseModel

from api.config import SETTINGS
from api.models import LLM_ENGINE
from api.utils.request import check_api_key

model_router = APIRouter # 创建一个名为model_router的API路由器对象


class ModelList(BaseModel):
    object: str = "list"  # 对象类型为列表
    data: List[Model] = []  # 包含Model对象的列表

available_models = ModelList(
    data=[
        Model(
            id=name,
            object="model",
            created=int(time.time()),
            owned_by="open"
        )
        for name in SETTINGS.model_names if name
    ]
)
"""
创建一个可用模型列表的实例，其中每个模型由Model对象表示，具有以下属性：
- id: 模型名称
- object: "model"
- created: 当前时间戳
- owned_by: "open"
"""

@model_router.get(
    "/models",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def show_available_models():
    """
    处理GET请求，返回可用模型列表。

    Returns:
        ModelList: 包含可用模型信息的ModelList对象。
    """
    res = available_models  # 获取可用模型列表实例
    exists = [m.id for m in res.data]  # 已存在的模型ID列表

    if SETTINGS.engine == "vllm":
        models = await LLM_ENGINE.show_available_models()  # 异步获取vLLM引擎可用模型列表
        for m in models.data:
            if m.id not in exists:
                res.data.append(m)  # 将新的模型添加到可用模型列表

    return res  # 返回包含可用模型信息的ModelList对象

@model_router.get(
    "/models/{model}",
    dependencies=[Depends(check_api_key)],
    status_code=status.HTTP_200_OK,
)
async def retrieve_model():
    """
    处理GET请求，检索特定模型的信息。

    Returns:
        Model: 包含特定模型信息的Model对象。
    """
    return Model(
        id=model,  # 模型名称
        object="model",  # 模型对象类型为"model"
        created=int(time.time()),  # 当前时间戳
        owned_by="open"  # 拥有者为"open"
    )

