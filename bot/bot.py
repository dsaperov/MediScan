import asyncio
import logging
import os

from aiogram import F
from aiogram import Bot, Dispatcher, types
from aiogram.utils.formatting import Text, Bold
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.feature import hog
from skimage.transform import rescale
import pandas as pd
import pickle

import cv2
from tensorflow import keras
from tensorflow.keras import layers, models

from utils import getenv_or_throw_exception

model = joblib.load("sgd_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("sc.pkl")
model_CNN = keras.models.load_model('cnn_model_v2')

modelMEL = pickle.load(open("LogRegForMEL.pkl", "rb"))

class_dict = {"MEL": 1, "NV": 2, "BCC": 3, "AKIEC": 4, "BKL": 5, "DF": 6, "VASC": 7}
dict_class = {ind-1: name for name, ind in class_dict.items()}

model_mode = "predict"

#ratings = []
#with open('ratings.pkl', 'wb') as f:
#    pickle.dump(ratings, f)

if not os.path.exists('ratings.pkl'):
    with open('ratings.pkl', 'wb') as f:
        pickle.dump([], f)


def get_ratings():
    try:
        with open('ratings.pkl', 'rb') as f:
            ratings = pickle.load(f)
    except EOFError:
        ratings = []
    return ratings


def predict_by_photo(bytesIO):
    image_vectors = []
    image = Image.open(bytesIO)
    gray_image = np.asarray(image.convert("L"))
    image = rescale(gray_image, 1 / 3, mode='reflect')
    img_hog, hog_img = hog(
        image, pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys')
    flat_vector = np.array(hog_img).flatten()
    image_vectors.append(flat_vector)
    image_vectors_array = np.array(image_vectors)
    image_vectors_array = pca.transform(image_vectors_array)
    df = pd.DataFrame(data=image_vectors_array)
    df = scaler.transform(df)
    prediction = model.predict(df)
    result = None
    for key, value in class_dict.items():
        if value == prediction:
            result = key
            break
    return result

def calc_square_means(img, val) :
    bot, top = 225-3*val, 225+3*val
    left, right = 300-4*val, 300+4*val
    sq = img[bot:top, left:right]
    return np.r_[sq.mean(axis=(0,1)), sq.std(axis=(0,1))]

def image_to_features(img):
    means = img.mean(axis=(0, 1))
    stds = img.std(axis=(0, 1))
    sq_res = calc_square_means(img, 40)
    return np.array([means[0], stds[0], sq_res[0], sq_res[3],
              means[1], stds[1], sq_res[1], sq_res[4],
              means[2], stds[2] ]).reshape(1,-1)

def predict_mel_by_photo(bytesIO) :
    image = Image.open(bytesIO)
    img = np.array(image)
    prob = modelMEL.predict_proba(image_to_features(img))
    return prob[0,1]

def predict_by_photo_CNN(bytesIO) :
    image = Image.open(bytesIO)
    img = np.array(image)
    img_resize = cv2.resize(img, None, fx=0.3, fy=0.3)
    prob = model_CNN.predict(np.array([img_resize]))
    return prob


logging.basicConfig(level=logging.INFO)
bot = Bot(token="6927350766:AAHYsGOciliUa_zmQjdBt6-_rOdnh-qxyLY")
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    content = Text(
        "Hello, ",
        Bold(message.from_user.full_name)
    )
    await message.answer(
        **content.as_kwargs()
    )
    await message.answer("Welcome to Mesidcan_hse_bot. For the list of possible commands use /help.")

@dp.message(Command("predict"))
async def cmd_predict(message: types.Message):
    global model_mode
    model_mode = "predict"
    await message.answer("Model operation mode changed to class prediction.")
    await message.answer("Please upload your photo.")

@dp.message(Command("predictmel"))
async def cmd_predictmel(message: types.Message):
    global model_mode
    model_mode = "predictmel"
    await message.answer("Model operation mode changed to predicting melanomas")
    await message.answer("Please upload your photo.")

@dp.message(Command("predictcnn"))
async def cmd_predictcnn(message: types.Message):
    global model_mode
    model_mode = "predictcnn"
    await message.answer("Model operation mode changed to predicting using CNN.")
    await message.answer("Please upload your photo.")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer("""Here is the list of possible commands: 
    /start - start 
    /help - get the list of all possible commands
    /predict - get the prediction which class is most likely
    /predictmel - get the probability that this is melanoma
    /predictcnn - get the class prediction using CNN 
    /rate - rate this bot
    /showrating - show bot rating                            
                """)

@dp.message(Command("rate"))
async def rate(message: types.Message):
    builder = InlineKeyboardBuilder()
    builder.add(types.InlineKeyboardButton(text="1", callback_data="1"))
    builder.add(types.InlineKeyboardButton(text="2", callback_data="2"))
    builder.add(types.InlineKeyboardButton(text="3", callback_data="3"))
    builder.add(types.InlineKeyboardButton(text="4", callback_data="4"))
    builder.add(types.InlineKeyboardButton(text="5", callback_data="5"))
    await message.answer("Choose a rating.", reply_markup=builder.as_markup())
    #rating = message.text[6:]
    #if rating in ["1", "2", "3", "4", "5"] :
    #    with open('ratings.pkl', 'rb') as f:
    #        ratings = pickle.load(f)
    #    ratings.append(int(rating))
    #    with open('ratings.pkl', 'wb') as f:
    #        pickle.dump(ratings, f)
    #    await message.answer("Your rating was counted.")
    #else :
    #    await message.answer("Invalid rating. Valid ratings are integers from 1 to 5. Examples: '/rate 1', '/rate 5'.")

@dp.callback_query(lambda F: F.data=="1" or F.data=="2" or F.data=="3" or F.data=="4" or F.data=="5")
async def add_rating(callback: types.CallbackQuery):
    rating = int(callback.data)
    ratings = get_ratings()
    ratings.append(int(rating))
    with open('ratings.pkl', 'wb') as f:
        pickle.dump(ratings, f)
    await callback.message.answer("Your rating was counted.")

@dp.message(Command("showrating"))
async def show_rate(message: types.Message):
    ratings = np.array(get_ratings())
    if ratings.shape[0] > 0 :
        await message.answer(f"Mean rating is {ratings.mean():.2f} using {ratings.shape[0]} rating entries.")
    else :
        await message.answer("No ratings given yet.")

@dp.message(Command("clearrating"))
async def clear_rate(message: types.Message):
    ratings = []
    with open('ratings.pkl', 'wb') as f:
        pickle.dump(ratings, f)
    await message.answer("Rating cleared.")

@dp.message(F.photo)
async def download_photo(message: types.Message, bot: Bot):
    await bot.download(message.photo[-1])
    photo_file = await bot.get_file(message.photo[-1].file_id)
    bytesIO = await bot.download_file(photo_file.file_path)
    if model_mode == "predict":
        pred = predict_by_photo(bytesIO)
        await message.answer(f"The most likely class is {pred}.")
    elif model_mode == "predictmel":
        prob = predict_mel_by_photo(bytesIO)
        await message.answer(f"The probability of it being a melanoma is {prob:.3f}.")
    else :
        prob = predict_by_photo_CNN(bytesIO)
        prediction = np.round(prob, decimals=5)
        dct = {"MEL": 0., "NV": 0., "BCC": 0., "AKIEC": 0., "BKL": 0., "DF": 0., "VASC": 0.}
        c = 0
        for key, value in dct.items():
            dct[key] = prediction[0][c]
            c += 1
        for key, value in dct.items():
            value = float("{:.2f}".format(value)) * 100
            await message.answer(f"The probability of {key} = {value} %")

@dp.message()
async def handle_unknown_command(message: types.Message, bot: Bot):
    await message.answer("Unknown command. For the list of possible commands use /help.")

async def main():
    await dp.start_polling(bot)

if __name__ =="__main__":
    asyncio.run(main())


