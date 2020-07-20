#!/usr/bin/env -S python -u

from __future__ import annotations
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, List
from sys import exit
from logging import getLogger, FileHandler, DEBUG, Formatter
from hashlib import sha3_256
from pathlib import Path
from base64 import b64decode
from json import dumps as json_dumps, loads as json_loads
from signal import signal, SIGTERM
from datetime import datetime

from pyutils.browser_extension_native_messaging import read_message, write_message
from pyutils.bloom_filter import BloomFilter
import cv2
import pytesseract
import numpy

from messages import RequestMessage, AddRequestMessage, QueryRequestMessage, QueryResponseMessage, QueryResponseEntry


LOG = getLogger(__name__)
LOG.setLevel(DEBUG)

SCREENSHOTS_DIRECTORY = Path('./screenshots')
ENTRY_LIST_PATH = Path('./entry_list.json')
STRING_ENCODING = 'utf-8'
IMG_HOST = 'http://localhost:4545'
LOG_PATH = './ocr_web_native_application.log'

file_handler = FileHandler(filename=LOG_PATH)
file_handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))
LOG.addHandler(hdlr=file_handler)


@dataclass
class Entry:
    url: str
    title: str
    timestamp_ms: int
    hash_value: bytes
    bloom_filter: BloomFilter

    @property
    def image_path(self) -> Path:
        return SCREENSHOTS_DIRECTORY / f'{self.timestamp_ms}_{self.hash_value.hex()}.png'


def custom_serializer(obj):
    if is_dataclass(obj):
        return asdict(obj)
    elif isinstance(obj, datetime):
        return str(datetime)
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, BloomFilter):
        return obj.export_bytes().hex()
    elif isinstance(obj, bytes):
        return obj.hex()
    else:
        raise TypeError(type(obj))


def run_listener(hash_to_entry: Dict[bytes, Entry]):

    LOG.debug('Starting listener.')

    english_stop_words = set(Path('resources/english_stop_words').read_text().splitlines())

    while message_bytes := read_message():

        LOG.debug(f'Read message bytes of size {len(message_bytes)} bytes.')

        message = RequestMessage.from_bytes(message_bytes=message_bytes)

        if isinstance(message, AddRequestMessage):
            image_data: bytes = b64decode(message.base64_img_data)
            image_hash: bytes = sha3_256(image_data).digest()

            if image_hash in hash_to_entry:
                LOG.debug(f'Skipped message for {message.url}.')
                continue

            encoded_words: List[bytes] = [
                word.encode(encoding=STRING_ENCODING)
                for word in set(
                    word.lower() for word in pytesseract.image_to_string(
                        image=cv2.imdecode(
                            buf=numpy.frombuffer(buffer=image_data, dtype=numpy.uint8),
                            flags=cv2.IMREAD_COLOR
                        ),
                        config='-l eng --oem 1 --psm 3'
                    ).split()
                ) - english_stop_words
            ]

            if not encoded_words:
                LOG.warning(f'No words in screenshot for {message.url}.')
                continue

            entry = Entry(
                url=message.url,
                title=message.title,
                timestamp_ms=message.timestamp_ms,
                hash_value=image_hash,
                bloom_filter=BloomFilter.from_values_2(values=encoded_words, capacity_proportion=1.0)
            )

            entry.image_path.write_bytes(image_data)
            hash_to_entry[image_hash] = entry

            LOG.info(f'Added entry for {message.url}')

        elif isinstance(message, QueryRequestMessage):
            write_message(
                message=json_dumps(
                    asdict(
                        QueryResponseMessage(
                            request_id=message.request_id,
                            entries=[
                                QueryResponseEntry(
                                    screenshot_src=f'{IMG_HOST}/{entry.image_path.name}',
                                    url=entry.url,
                                    title=entry.title,
                                    timestamp_ms=entry.timestamp_ms
                                )
                                for entry in hash_to_entry.values()
                                for query_word in set(query_word.lower() for query_word in message.query.split())
                                if query_word.encode(encoding=STRING_ENCODING) in entry.bloom_filter
                            ]
                        )

                    )
                ).encode(encoding=STRING_ENCODING)
            )
        else:
            LOG.error(f'Unsupported message type: {message.__class__}')


def main():

    hash_to_entry: Dict[bytes, Entry] = {}
    try:
        hash_to_entry = {
            bytes.fromhex(json_entry['hash_value']): Entry(
                url=json_entry['url'],
                title=json_entry['title'],
                timestamp_ms=json_entry['timestamp_ms'],
                hash_value=bytes.fromhex(json_entry['hash_value']),
                bloom_filter=BloomFilter.import_bytes(data=bytes.fromhex(json_entry['bloom_filter']))
            )
            for json_entry in json_loads(ENTRY_LIST_PATH.read_bytes().decode())
        }
    except OSError as err:
        LOG.warning(f'OSError deserializing: {err}')
        pass

    def save_hash_to_entry(_=None, __=None) -> None:
        ENTRY_LIST_PATH.write_bytes(
            data=json_dumps(
                list(hash_to_entry.values()),
                default=custom_serializer
            ).encode()
        )

    signal(SIGTERM, save_hash_to_entry)

    run_listener(hash_to_entry=hash_to_entry)

    save_hash_to_entry()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        LOG.exception(e)
        exit(1)
