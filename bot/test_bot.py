import random
import string

import pytest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler, CallbackQueryHandler
from aiogram_tests.types.dataset import MESSAGE, CALLBACK_QUERY

from bot import cmd_help
from bot import cmd_start
from bot import handle_unknown_command
from bot import rate
from bot import add_rating
from bot import clear_rate

strings = string.ascii_letters + string.digits
random_string = ''.join(random.choice(strings) for _ in range(7))


class TestCallbackData(CallbackData, prefix="callback_data"):
    rate: str


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


# @pytest.mark.asyncio
# async def test_add_rating():
#     requester = MockedBot(CallbackQueryHandler(add_rating,TestCallbackData.filter()))
#     calls = await requester.query(CALLBACK_QUERY.as_object(data="5"))
#     answer_message = calls.send_message.fetchone().text
#     assert answer_message == "Your rating was counted."
