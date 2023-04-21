import telebot
import os 
from telebot import types
import requests
import recognition
from PIL import Image 

bot = telebot.TeleBot(os.environ['TOKEN']) 
model = recognition.load_model('./savemodel/best_model.pth')

@bot.message_handler(content_types= ["photo"])
def verifyUser(message):
    print(len(message.photo))
    file = bot.get_file(message.photo[-1].file_id)
    print(file)
    photo = bot.download_file(file.file_path)
    with open("image.jpg", 'wb') as new_file:
        new_file.write(photo)
    from PIL import Image

    im1 = Image.open(r'image.jpg')
    im1.save(r'image.png')
    labels = recognition.look_for_helmets(model, "image.png", 0.7)
    bot.send_message(message.chat.id, "Labels: "+ str(labels) )

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('How to use')
    btn2 = types.KeyboardButton('Knowledge Base')
    btn3 = types.KeyboardButton('Training Notebook')

    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "👋 Hi! I'm SafetyHelmetDetector bot!", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == 'How to use':
        bot.send_message(message.from_user.id, "We'll define some best practices a bit later", parse_mode='Markdown')
    elif message.text == 'Knowledge Base':
        reply = "The latest version of the project's Knowledge is [here](https://github.com/kuroyuki/SafetyHelmetDetection/blob/main/knowedge)"
        bot.send_message(message.from_user.id, reply, parse_mode='Markdown', disable_web_page_preview=True)
    elif message.text == 'Training Notebook':
        reply = "Please checkout [here](https://github.com/kuroyuki/SafetyHelmetDetection/blob/main/src/training.ipynb)"
        bot.send_message(message.from_user.id, reply, parse_mode='Markdown', disable_web_page_preview=True)
    else :
        bot.send_message(message.from_user.id,"We've got your message but have no idea at the moment what to do with it. \nSorry", parse_mode='Markdown')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть

