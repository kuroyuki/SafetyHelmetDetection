import telebot
import os 
from telebot import types
import requests
bot = telebot.TeleBot(os.environ['TOKEN']) 

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
        reply = requests.get("https://raw.githubusercontent.com/kuroyuki/SafetyHelmetDetection/main/knowledge.base.n3")
        bot.send_message(message.from_user.id, reply, parse_mode='Markdown')
    elif message.text == 'Training Notebook':
        reply = requests.get("https://raw.githubusercontent.com/kuroyuki/SafetyHelmetDetection/main/src/training.ipynb")
        bot.send_message(message.from_user.id, reply, parse_mode='Markdown')
    else :
        bot.send_message(message.from_user.id,"We've got your message but have no idea at the moment what to do with it. \nSorry", parse_mode='Markdown')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть