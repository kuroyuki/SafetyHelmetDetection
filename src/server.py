import telebot
import os 
from telebot import types
import requests
import recognition
from PIL import Image, ImageDraw
import rdflib

bot = telebot.TeleBot(os.environ['TOKEN']) 
model = recognition.load_model('./savemodel/best_model_vitaliy.pth')
# yolo_model = recognition.load_yolo5('./savemodel/best_model_yolo.pt')
selectedModel = "Model 2"

#Load initial KPIs
g = rdflib.Graph()
result = g.parse("https://raw.githubusercontent.com/kuroyuki/SafetyHelmetDetection/main/knowledge/KB_SHD.n3", format="text/n3")

min_threshold = 0.7 #defaut value
qres = g.query(
    """SELECT DISTINCT ?class ?min_threshold
       WHERE {
          ?class a classes:KPI .
          ?class rdfs:label "Threshold" . 
          ?class prop:hasMinValue  ?min_threshold .
       }""")

for row in qres:
    min_threshold = float(row.asdict()['min_threshold'].toPython())

dimension = 416 #defaut value 
qres = g.query(
    """SELECT DISTINCT ?class ?dimension
       WHERE {
          ?class a classes:KPI .
          ?class rdfs:label "Image dimensions" . 
          ?class prop:hasMinValue  ?dimension .
       }""")

for row in qres:
    dimension = int(row.asdict()['dimension'].toPython())

print(dimension, min_threshold)

def preprocess_input_image(img):
    return img.resize((dimension, dimension))

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
    [labels, boxes] = recognition.look_for_helmets(model, "image.png", min_threshold)

    #prepare answer
    caption = "No violations detected"
    if "head" in labels:
        caption="Warning !!!!"
    if len(labels) == 0:
        caption="Warning !!!!"
    #add boxes
    output_image = prepare_output_image(input_image, labels, boxes)
    #add red frame to the image 
    global dimension
    if caption != "No violations detected":
        draw = ImageDraw.Draw(output_image)
        color = 'red'
        draw.line((2,2, dimension-2, 2), fill=color, width=2)
        draw.line((dimension-2,2, dimension-2, dimension-2), fill=color, width=2)
        draw.line((dimension-2, dimension-2, 2, dimension-2), fill=color, width=2)
        draw.line((2, dimension-2, 2,2), fill=color, width=2)


    bot.send_photo(message.chat.id, output_image, caption=(caption+str(labels)+str(boxes)))

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Model 1')
    btn2 = types.KeyboardButton('Model 2')
    btn3 = types.KeyboardButton('Model 3')

    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "👋 Hi! I'm SafetyHelmetDetector bot!\nYou can choose one of our models to start detection of the helmets on your photos.\n Model 2 is default choice\n Good luck ", reply_markup=markup)

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