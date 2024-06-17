from __future__ import annotations

from typing import Any, cast, Dict, Type

import pydantic

# --------------- Pydantic v2 compatibility ---------------

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")  # 检查是否为Pydantic v2版本

def model_json(model: pydantic.BaseModel, **kwargs) -> str:
    """
    返回模型的JSON表示形式。
    """
    if PYDANTIC_V2:
        return model.model_dump_json(**kwargs)  # 如果是Pydantic v2，返回模型的JSON表示形式
    return model.json(**kwargs)  # 否则返回模型的JSON字符串表示形式

def model_dump(model: pydantic.BaseModel, **kwargs) -> Dict[str, Any]:
    """
    返回模型的字典表示形式。
    """
    if PYDANTIC_V2:
        return model.model_dump(**kwargs)  # 如果是Pydantic v2，返回模型的字典表示形式
    return cast(
        "dict[str, Any]",
        model.dict(**kwargs),
    )  # 否则返回模型的字典形式

def model_parse(model: Type[pydantic.BaseModel], data: Any) -> pydantic.BaseModel:
    """
    解析给定数据到模型实例。
    """
    if PYDANTIC_V2:
        return model.model_validate(data)  # 如果是Pydantic v2，进行模型验证并返回
    return model.parse_obj(data)  # 否则解析数据为模型对象

def disable_warnings(model: Type[pydantic.BaseModel]):
    """
    禁用与模型名称设置相关的警告。
    """
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()  # 如果是Pydantic v2，设置模型配置中的protected_namespaces为空元组

