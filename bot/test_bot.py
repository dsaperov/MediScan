import random
import string
from io import BytesIO
from unittest.mock import patch

import pytest
from aiogram import Bot
from aiogram.filters import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE, MESSAGE_WITH_PHOTO

from bot import cmd_help
from bot import cmd_start
from bot import handle_unknown_command
from bot import rate
from bot import clear_rate
from bot import download_photo
from bot import cmd_predict


strings = string.ascii_letters + string.digits
random_string = ''.join(random.choice(strings) for _ in range(7))


async def mock_download(bot, message):
    with open("/Users/aleksey/PycharmProjects/MediScan/bot/ISIC_0034323.jpg", "rb") as f:
        photo_data = f.read()
    bytesio_object = BytesIO(photo_data)
    return bytesio_object


@pytest.mark.asyncio
async def test_cmd_help():
    requester = MockedBot(MessageHandler(cmd_help, Command(commands=["help"])))
    calls = await requester.query(MESSAGE.as_object(text="/help"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == """Here is the list of possible commands: 
    /start - start 
    /help - get the list of all possible commands
    /predict - get the prediction which class is most likely
    /predictmel - get the probability that this is melanoma
    /predictcnn - get the class prediction using CNN 
    /rate - rate this bot
    /showrating - show bot rating                            
                """


@pytest.mark.asyncio
async def test_cmd_start():
    requester = MockedBot(MessageHandler(cmd_start, Command(commands=["start"])))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Welcome to Mesidcan_hse_bot. For the list of possible commands use /help."


@pytest.mark.asyncio
async def test_handle_unknown_command():
    requester = MockedBot(MessageHandler(handle_unknown_command))
    calls = await requester.query(MESSAGE.as_object(text=random_string))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Unknown command. For the list of possible commands use /help."


@pytest.mark.asyncio
async def test_rate():
    requester = MockedBot(MessageHandler(rate))
    calls = await requester.query(MESSAGE.as_object(text="/rate"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Choose a rating."

@pytest.mark.asyncio
async def test_clear_rate():
    requester = MockedBot(MessageHandler(clear_rate))
    calls = await requester.query(MESSAGE.as_object(text="/clearrating"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Rating cleared."


@patch.object(Bot, 'download', mock_download, create=True)
@pytest.mark.asyncio
async def test_cmd_predict():
    requester = MockedBot(MessageHandler(cmd_predict))
    calls = await requester.query(MESSAGE.as_object(text="/predict"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Please upload your photo."

    requester = MockedBot(MessageHandler(download_photo))
    calls = await requester.query(MESSAGE_WITH_PHOTO.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "The most likely class is NV."
