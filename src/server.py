import telebot
import os 
from telebot import types

bot = telebot.TeleBot(os.environ['TOKEN']) 

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('How to use')
    markup.add(btn1)
    bot.send_message(message.from_user.id, "👋 Hi! I'm SafetyHelmetDetector bot!", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == 'How to use':
        bot.send_message(message.from_user.id, "We'll define some best practices a bit later", parse_mode='Markdown')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть