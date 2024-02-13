import random
import string

import pytest
from aiogram.filters import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE

from bot import cmd_help
from bot import cmd_start
from bot import handle_unknown_command

strings = string.ascii_letters + string.digits
random_string = ''.join(random.choice(strings) for _ in range(7))


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
