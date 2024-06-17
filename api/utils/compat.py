from __future__ import annotations

from typing import Any, Dict, Type

import pydantic
from pydantic import BaseModel

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")

# 根据数据模型版本，将数据模型转换为字典
def dictify(data: "BaseModel", **kwargs) -> Dict[str, Any]:
    try:  # 尝试使用pydantic v2的方法
        return data.model_dump(**kwargs)
    except AttributeError:  # 如果不存在model_dump方法，则使用pydantic v1的方法
        return data.dict(**kwargs)

# 根据数据模型版本，将数据模型转换为JSON字符串
def jsonify(data: "BaseModel", **kwargs) -> str:
    try:  # 尝试使用pydantic v2的方法
        return data.model_dump_json(**kwargs)
    except AttributeError:  # 如果不存在model_dump_json方法，则使用pydantic v1的方法
        return data.json(**kwargs)

# 根据数据模型版本，对给定对象进行验证并返回验证后的数据模型对象
def model_validate(data: Type["BaseModel"], obj: Any) -> "BaseModel":
    try:  # 尝试使用pydantic v2的方法
        return data.model_validate(obj)
    except AttributeError:  # 如果不存在model_validate方法，则使用pydantic v1的parse_obj方法
        return data.parse_obj(obj)

# 禁用模型警告
def disable_warnings(model: Type["BaseModel"]):
    # 禁用model_name设置的警告
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()

