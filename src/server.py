import telebot
import os 
from telebot import types
import requests
import recognition
from PIL import Image 

bot = telebot.TeleBot(os.environ['TOKEN']) 
model = recognition.load_model('./savemodel/best_model_andrew.pth')


def preprocess_input_image(img):
    return img.resize((416, 416))

def prepare_output_image(img, labels, boxes):
    return img

@bot.message_handler(content_types= ["photo"])
def verifyUser(message):
    #Get image from telegram
    file = bot.get_file(message.photo[-1].file_id)
    photo = bot.download_file(file.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(photo)
    input_image = Image.open(r'image.jpg')

    #prepare image for further processing 
    input_image = preprocess_input_image(input_image)
    input_image.save(r'image.png')

    #recognise 
    labels = recognition.look_for_helmets(model, "image.png", 0.7)

    #prepare answer
    caption = "No violations detected"
    if "head" in labels:
        caption="Warning !!!!"
    bot.send_photo(message.chat.id, prepare_output_image(input_image), caption=caption)

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Model 1')
    btn2 = types.KeyboardButton('Model 2')
    btn3 = types.KeyboardButton('Model 3')

    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "👋 Hi! I'm SafetyHelmetDetector bot!\nYou can choose one of our models to start detection of the helmets on your photos.\n Model 3 is default choice\n Good luck ", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global model
    if message.text == 'Model 1':
        model = recognition.load_model('./savemodel/best_model.pth')
        bot.send_message(message.from_user.id, "Using Model 1", parse_mode='Markdown')
    elif message.text == 'Model 2':
        model = recognition.load_model('./savemodel/best_model_vitaliy.pth')
        bot.send_message(message.from_user.id, "Using Model 2", parse_mode='Markdown')
    elif message.text == 'Model 3':
        model = recognition.load_model('./savemodel/best_model_andrew.pth')
        bot.send_message(message.from_user.id, "Using Model 3", parse_mode='Markdown')
    else :
        bot.send_message(message.from_user.id,"We've got your message but have no idea at the moment what to do with it. \nSorry", parse_mode='Markdown')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть

