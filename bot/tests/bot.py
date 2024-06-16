import asyncio
import logging

from aiogram import F
from aiogram import Bot, Dispatcher, types
from aiogram.utils.formatting import Text, Bold
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.transform import rescale
import pandas as pd
import pickle
import torch
import torchvision.transforms as transforms

# import cv2
# from tensorflow import keras
from database import save_rating, get_ratings, delete_rating, check_rating
from utils import getenv_or_throw_exception, is_running_in_docker, get_docker_secret

TOKEN = "BOT_TOKEN"

model = joblib.load("sgd_model.pkl")
pca = joblib.load("pca.pkl")
scaler = joblib.load("sc.pkl")
# model_CNN = keras.models.load_model('cnn_model_v2')
target_size = (600, 450)

modelMEL = pickle.load(open("LogRegForMEL.pkl", "rb"))
modelENB0 = torch.load('efficient_net_b4/model_base.pth',
                       map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

class_dict = {"MEL": 1, "NV": 2, "BCC": 3,
              "AKIEC": 4, "BKL": 5, "DF": 6, "VASC": 7}
dict_class = {ind - 1: name for name, ind in class_dict.items()}

model_mode = "predict"


def predict_by_photo_cnn(bytesIO):
    image = Image.open(bytesIO).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_image = transform(image)

    input_image = input_image.unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = input_image.to(device)

    with torch.no_grad():
        output = modelENB0(input_image)

    predicted_class = torch.argmax(output, dim=1).item()

    return dict_class[predicted_class]


def predict_by_photo(bytesIO):
    image_vectors = []
    image = Image.open(bytesIO)
    width, height = image.size
    if (width, height) != target_size:
        image = image.resize(target_size)
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


def calc_square_means(img, val):
    bot, top = 225 - 3 * val, 225 + 3 * val
    left, right = 300 - 4 * val, 300 + 4 * val
    sq = img[bot:top, left:right]
    return np.r_[sq.mean(axis=(0, 1)), sq.std(axis=(0, 1))]


def image_to_features(img):
    means = img.mean(axis=(0, 1))
    stds = img.std(axis=(0, 1))
    sq_res = calc_square_means(img, 40)
    return np.array([means[0], stds[0], sq_res[0], sq_res[3],
                     means[1], stds[1], sq_res[1], sq_res[4],
                     means[2], stds[2]]).reshape(1, -1)


def predict_mel_by_photo(bytesIO):
    image = Image.open(bytesIO)
    img = np.array(image)
    prob = modelMEL.predict_proba(image_to_features(img))
    return prob[0, 1]


# def predict_by_photo_CNN(bytesIO):
#     image = Image.open(bytesIO)
#     width, height = image.size
#     if (width, height) != target_size:
#         image = image.resize(target_size)
#     img = np.array(image)
#     img_resize = cv2.resize(img, None, fx=0.3, fy=0.3)
#     prob = model_CNN.predict(np.array([img_resize]))
#     return prob


logging.basicConfig(level=logging.INFO)

if is_running_in_docker():
    token = get_docker_secret(TOKEN)
else:
    token = getenv_or_throw_exception(TOKEN)

bot = Bot(token=token)
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
    await message.answer("Welcome to Mesidcan_hse_bot. "
                         "For the list of possible commands use /help.")


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


# @dp.message(Command("predictcnn"))
# async def cmd_predictcnn(message: types.Message):
#     global model_mode
#     model_mode = "predictcnn"
#     await message.answer("Model operation mode changed to predicting using CNN.")
#     await message.answer("Please upload your photo.")


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
    /predictcnn - get the class prediction using CNN (efficient_net_b0)
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


@dp.callback_query(lambda F: F.data in ["1", "2", "3", "4", "5"])
async def add_rating(callback: types.CallbackQuery, bot: Bot):
    message = callback.message
    user_id = callback.from_user.id
    rating = int(callback.data)
    save_rating(user_id, rating)
    await bot.edit_message_text(text="Your rate is counted.", message_id=message.message_id, chat_id=message.chat.id)


@dp.message(Command("showrating"))
async def show_rate(message: types.Message):
    ratings = get_ratings()
    ratings_num = len(ratings)
    mean_rating = sum(ratings) / ratings_num if ratings_num > 0 else 0
    if ratings_num > 0:
        await message.answer(f"Mean rating is {mean_rating:.2f} using {ratings_num} rating entries.")
    else:
        await message.answer("No ratings given yet.")


@dp.message(Command("clearrating"))
async def clear_rate(message: types.Message):
    user_id = message.from_user.id
    user_sent_rating = check_rating(user_id)
    if user_sent_rating:
        delete_rating(user_id)
        await message.answer("Rating cleared.")
    else:
        await message.answer("No rating to clear.")


@dp.message(F.photo)
async def download_photo(message: types.Message, bot: Bot):
    bytesIO = await bot.download(message.photo[-1].file_id)
    if model_mode == "predict":
        pred = predict_by_photo(bytesIO)
        await message.answer(f"The most likely class is {pred}.")
    elif model_mode == "predictmel":
        prob = predict_mel_by_photo(bytesIO)
        await message.answer(f"The probability of it being a melanoma is {prob:.3f}.")
    # elif model_mode == "predictcnn":
    #     prob = predict_by_photo_CNN(bytesIO)
    #     prediction = np.round(prob, decimals=5)
    #     dct = {"MEL": 0., "NV": 0., "BCC": 0., "AKIEC": 0., "BKL": 0., "DF": 0., "VASC": 0.}
    #     c = 0
    #     for key, value in dct.items():
    #         dct[key] = prediction[0][c]
    #         c += 1
    #     for key, value in dct.items():
    #         value = float("{:.2f}".format(value)) * 100
    #         await message.answer(f"The probability of {key} = {value} %")
    else:
        pred = predict_by_photo_cnn(bytesIO)
        await message.answer(f"The most likely class is {pred}.")


@dp.message()
async def handle_unknown_command(message: types.Message, bot: Bot):
    await message.answer("Unknown command. For the list of possible commands use /help.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
