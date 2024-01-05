import asyncio
import logging
from aiogram import F
from aiogram import Bot, Dispatcher, types
from aiogram.utils.formatting import Text, Bold
from aiogram.filters.command import Command
import joblib
import numpy as np
from PIL import Image
from io import BytesIO
from skimage.feature import hog
from skimage.transform import rescale
import pandas as pd
import pickle

model = joblib.load("sgd_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("sc.pkl")

modelMEL = pickle.load(open("LogRegForMEL.pkl", "rb"))

class_dict = {"MEL": 1, "NV": 2, "BCC": 3, "AKIEC": 4, "BKL": 5, "DF": 6, "VASC": 7}

model_mode = "predict"

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


logging.basicConfig(level=logging.INFO)
bot = Bot(token = "6927350766:AAHYsGOciliUa_zmQjdBt6-_rOdnh-qxyLY")
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
    await message.answer("Welcome to Mesidcan_hse_bot. For the list of possible commands use /help")
@dp.message(Command("predict"))
async def cmd_start(message: types.Message):
    global model_mode
    model_mode = "predict"
    await message.answer("Model operation mode changed to class prediction.")
    await message.answer("Please download your photo")

@dp.message(Command("predictmel"))
async def cmd_start(message: types.Message):
    global model_mode
    model_mode = "predictmel"
    await message.answer("Model operation mode changed to predicting melanomas")
    await message.answer("Please download your photo")

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    await message.answer("""Here is the list of possible commands: 
    /start - start 
    /help - get the list of all possible commands
    /predict - get the prediction which class is most likely
    /predictmel -get the probability that this is melanoma
    
                                
                """)

@dp.message(F.photo)
async def download_photo(message: types.Message, bot: Bot):
    await bot.download(
        message.photo[-1],
        #destination = f"./tmp/{message.photo[-1].file_id}.jpg"
    )
    #img_path = (await bot.get_file(message.photo[-1].file_id)).file_path
    photo_file = await bot.get_file(message.photo[-1].file_id)
    bytesIO = await bot.download_file(photo_file.file_path)
    if model_mode == "predict":
        pred = predict_by_photo(bytesIO)
        await message.answer(f"The most likely class is {pred}.")
    else :
        prob = predict_mel_by_photo(bytesIO)
        await message.answer(f"The probability of it being a melanoma is {prob:.3f}.")


async def main():
    await dp.start_polling(bot)

if __name__ =="__main__":
    asyncio.run(main())


