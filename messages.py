from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, ClassVar, List
from json import loads as json_loads
from abc import ABC


@dataclass
class Message(ABC):
    request_id: int


@dataclass
class RequestMessage(Message, ABC):

    MESSAGE_TYPE: ClassVar[str] = NotImplemented

    @classmethod
    def from_bytes(cls, message_bytes: bytes) -> RequestMessage:
        message_dict: Dict[str, Any] = json_loads(message_bytes)
        message_type: str = message_dict.pop('type')

        if message_type == AddRequestMessage.MESSAGE_TYPE:
            return AddRequestMessage(**message_dict)
        elif message_type == QueryRequestMessage.MESSAGE_TYPE:
            return QueryRequestMessage(**message_dict)
        else:
            raise ValueError(f'Not a supported message type: {message_type}.')


@dataclass
class AddRequestMessage(RequestMessage):
    MESSAGE_TYPE: ClassVar[str] = 'add'

    base64_img_data: str
    url: str
    title: str
    timestamp_ms: int


@dataclass
class QueryRequestMessage(RequestMessage):
    MESSAGE_TYPE: ClassVar[str] = 'query'

    query: str


@dataclass
class ResponseMessage(Message, ABC):
    pass


@dataclass
class QueryResponseEntry:
    screenshot_src: str
    url: str
    title: str
    timestamp_ms: int


@dataclass
class QueryResponseMessage(ResponseMessage):
    entries: List[QueryResponseEntry]
