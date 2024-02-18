import os
import random
import string
from io import BytesIO
from unittest.mock import patch, MagicMock
import pickle

import pytest
from aiogram import Bot
from aiogram.filters import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE, MESSAGE_WITH_PHOTO

from bot import cmd_help
from bot import cmd_start
from bot import handle_unknown_command
from bot import clear_rate
from bot import download_photo
from bot import cmd_predict
from bot import cmd_predictcnn
from bot import cmd_predictmel
from bot import show_rate
from bot import rate
from bot import add_rating
from bot import get_ratings

strings = string.ascii_letters + string.digits
random_string = ''.join(random.choice(strings) for _ in range(7))


async def mock_download(bot, message):
    with open("ISIC_0034323.jpg", "rb") as f:
        photo_data = f.read()
    bytesio_object = BytesIO(photo_data)
    return bytesio_object


async def mock_add_rating(callback, bot):
    rating = int(callback.data)
    ratings = get_ratings()
    ratings.append(int(rating))
    with open('ratings.pkl', 'wb') as f:
        pickle.dump(ratings, f)
    await bot.edit_message_text(text="Your rating was counted.",
                                chat_id=callback.message.chat.id,
                                message_id=callback.message.message_id)


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
    requester = MockedBot(MessageHandler(cmd_start,
                                         Command(commands=["start"])))
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


@patch.object(Bot, 'download', mock_download, create=True)
@pytest.mark.asyncio
async def test_predict():
    requester = MockedBot(MessageHandler(cmd_predict))
    calls = await requester.query(MESSAGE.as_object(text="/predict"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Please upload your photo."

    requester = MockedBot(MessageHandler(download_photo))
    calls = await requester.query(MESSAGE_WITH_PHOTO.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "The most likely class is NV."

    requester = MockedBot(MessageHandler(cmd_predictcnn))
    calls = await requester.query(MESSAGE.as_object(text="/predictcnn"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Please upload your photo."

    requester = MockedBot(MessageHandler(download_photo))
    calls = await requester.query(MESSAGE_WITH_PHOTO.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "The probability of VASC = 6.0 %"

    requester = MockedBot(MessageHandler(cmd_predictmel))
    calls = await requester.query(MESSAGE.as_object(text="/predictmel"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Please upload your photo."

    requester = MockedBot(MessageHandler(download_photo))
    calls = await requester.query(MESSAGE_WITH_PHOTO.as_object())
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "The probability of it being a melanoma is 0.039."


@pytest.mark.asyncio
async def test_rating():
    requester = MockedBot(MessageHandler(clear_rate))
    calls = await requester.query(MESSAGE.as_object(text="/clearrating"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Rating cleared."

    requester = MockedBot(MessageHandler(show_rate))
    calls = await requester.query(MESSAGE.as_object(text="/showrate"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "No ratings given yet."

    requester = MockedBot(MessageHandler(rate))
    calls = await requester.query(MESSAGE.as_object(text="/rate"))
    answer_message = calls.send_message.fetchone().text
    assert answer_message == "Choose a rating."


@pytest.mark.asyncio
async def test_add_rating():
    mock_callback_query = MagicMock(data="3")
    mock_bot = MagicMock(spec=Bot)
    mock_message = MagicMock()
    mock_callback_query.message = mock_message

    with (patch('bot.get_ratings', return_value=[]),
          patch('builtins.open', create=True),
          patch('pickle.dump'),
          patch.object(mock_bot, 'edit_message_text') as mock_edit_message_text):
        await add_rating(mock_callback_query, mock_bot)

        mock_edit_message_text \
            .assert_called_once_with(text="Your rate is counted.",
                                     message_id=mock_message.message_id,
                                     chat_id=mock_message.chat.id)

        requester = MockedBot(MessageHandler(show_rate))
        calls = await requester.query(MESSAGE.as_object(text="/showrate"))
        answer_message = calls.send_message.fetchone().text
        assert answer_message == "Mean rating is 3.00 using 1 rating entries."


def test_ratings_file_creation(monkeypatch):
    """Test that the ratings file is created if it doesn't exist."""
    ratings_file = "ratings.pkl"
    ratings_file_renamed = "ratings.pkl.bak"

    if os.path.exists(ratings_file):
        os.rename(ratings_file, ratings_file_renamed)

    try:
        exec(open("bot.py").read())
        assert os.path.exists(ratings_file)
        if os.path.exists(ratings_file):
            os.remove(ratings_file)
    finally:
        if os.path.exists(ratings_file_renamed):
            os.rename(ratings_file_renamed, ratings_file)
