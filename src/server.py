import telebot
import os 
from telebot import types
import requests
import recognition
from PIL import Image, ImageDraw
import rdflib
import time

bot = telebot.TeleBot(os.environ['TOKEN']) 

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
    width, height = img.size
    # Create a new blank square image with a white background
    square_image = Image.new('RGB', (dimension, dimension), (255, 255, 255))

    # Calculate the offset to center the original image
    x_offset = (dimension - width) // 2
    y_offset = (dimension - height) // 2

    # Paste the original image onto the square image
    square_image.paste(img, (x_offset, y_offset))
    
    return square_image

def prepare_output_image(img, labels, boxes):
    draw = ImageDraw.Draw(img)
    for index, box in enumerate(boxes):
        color = 'lightgreen'
        width =2 
        if labels[index] != "helmet":
            color = "red"
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

    start_time = time.time()

    global dimension

    #recognise 
    [labels, boxes] = recognition.find_helmets("image.png", min_threshold, dimension)

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
        draw.line((2,2, dimension-2, 2), fill=color, width=2)
        draw.line((dimension-2,2, dimension-2, dimension-2), fill=color, width=2)
        draw.line((dimension-2, dimension-2, 2, dimension-2), fill=color, width=2)
        draw.line((2, dimension-2, 2,2), fill=color, width=2)

    print("--- %s seconds ---" % (time.time() - start_time))

    bot.send_photo(message.chat.id, output_image, caption=caption + " Took "+str((time.time() - start_time))+" s")

@bot.message_handler(commands=['start'])
def start(message):

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton('Yolo5')
    btn2 = types.KeyboardButton('CNN 1')
    btn3 = types.KeyboardButton('CNN 2')

    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "👋 Hi! I'm SafetyHelmetDetector bot!\nYou can choose one of our models to start detection of the helmets on your photos.\n Yolo5 is default choice\n Good luck ", reply_markup=markup)

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    recognition.load_model(message.text)
    bot.send_message(message.from_user.id, "Using "+str(message.text), parse_mode='Markdown')
  

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть
