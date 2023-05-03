import telebot
import os 
from telebot import types
import requests
import recognition
from PIL import Image, ImageDraw

bot = telebot.TeleBot(os.environ['TOKEN']) 
model = recognition.load_model('./savemodel/best_model_vitaliy.pth')
yolo_model = recognition.load_yolo5('./savemodel/best_model_yolo.pth')
selectedModel = "Model 2"

def preprocess_input_image(img):
    return img.resize((416, 416))

def prepare_output_image(img, labels, boxes):
    draw = ImageDraw.Draw(img)
    for index, box in enumerate(boxes):
        color = 'lightgreen'
        width =1 
        if labels[index] != "helmet":
            color = "red"
            width = 2
        draw.line((box[0], box[1], box[2], box[1]), fill=color, width=width)
        draw.line((box[2], box[1], box[2], box[3]), fill=color, width=width)
        draw.line((box[2], box[3], box[0], box[3]), fill=color, width=width)
        draw.line((box[0], box[3], box[0], box[1]), fill=color, width=width)
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
    [labels, boxes]
    if selectedModel == 'Model 1':
        [labels, boxes] = recognition.look_for_helmets_with_yolo(yolo_model, "image.png")
    else:
        [labels, boxes] = recognition.look_for_helmets(model, "image.png", 0.7)

    #prepare answer
    caption = "No violations detected"
    if "head" in labels:
        caption="Warning !!!!"
    if len(labels) == 0:
        caption="Warning !!!!"
    #add boxes
    output_image = prepare_output_image(input_image, labels, boxes)
    #add red frame to the image 
    if caption != "No violations detected":
        draw = ImageDraw.Draw(output_image)
        color = 'red'
        draw.line((2,2, 414, 2), fill=color, width=2)
        draw.line((414,2, 414, 414), fill=color, width=2)
        draw.line((414, 414, 2, 414), fill=color, width=2)
        draw.line((2, 414, 2,2), fill=color, width=2)


    bot.send_photo(message.chat.id, output_image, caption=(caption+str(labels)+str(boxes)))

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Model 1')
    btn2 = types.KeyboardButton('Model 2')
    btn3 = types.KeyboardButton('Model 3')

    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "👋 Hi! I'm SafetyHelmetDetector bot!\nYou can choose one of our models to start detection of the helmets on your photos.\n "+selectedModel+" is default choice\n Good luck ", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    global model, selectedModel
    if message.text == 'Model 1':
        model = recognition.load_model('./savemodel/best_model.pth')
        selectedModel = 'Model 1'
        bot.send_message(message.from_user.id, "Using Model 1", parse_mode='Markdown')
    elif message.text == 'Model 2':
        model = recognition.load_model('./savemodel/best_model_vitaliy.pth')
        selectedModel = 'Model 2'
        bot.send_message(message.from_user.id, "Using Model 2", parse_mode='Markdown')
    elif message.text == 'Model 3':
        model = recognition.load_model('./savemodel/best_model_andrew.pth')
        selectedModel = 'Model 3'
        bot.send_message(message.from_user.id, "Using Model 3", parse_mode='Markdown')
    else :
        bot.send_message(message.from_user.id,"We've got your message but have no idea at the moment what to do with it. \nSorry", parse_mode='Markdown')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть

