import cv2
from pytesseract import image_to_string
from PIL import Image
import numpy as np
import os
import jamspell
from matplotlib import pyplot as plt
import re
from skimage import img_as_ubyte
from skimage import data
from skimage.exposure import histogram
from skimage.filters.rank import median
from skimage.filters.rank import mean_bilateral
from skimage.morphology import disk
from skimage.filters.rank import otsu
from skimage.filters import threshold_otsu
import math
import dlib
import imutils


input_path = r'/home/paul/Desktop/Passport_detect/images/6.jpg'
#input_path = r'/home/paul/Desktop/Passport_detect/text-detection-ctpn/data/p_9.jpg'
output_path = r'/home/paul/Desktop/Passport_detect/result'

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

def get_images(in_path):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(in_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def get_text(in_path):
    files = []
    exts = ['txt']
    for parent, dirnames, textnames in os.walk(in_path):
        for filename in textnames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def crop_first_page(img, scale_x=1.0, scale_y = 1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale_x, img.shape[0] * scale_y
    left_x, right_x = center_x - width_scaled/2 , center_x + width_scaled/2
    top_y, bottom_y = center_y - height_scaled*1.5 , center_y - height_scaled/1.5
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def crop_second_page(img, scale_x=1.0, scale_y = 1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale_x, img.shape[0] * scale_y
    left_x, right_x = center_x - width_scaled*1.4 , center_x + width_scaled/2
    top_y, bottom_y = center_y, center_y + height_scaled*1.2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def crop_num(img, scale_x=1.0, scale_y=1.0):
    center_x, center_y = img.shape[1]/2, img.shape[0]/2
    width_scaled, height_scaled = img.shape[1]*scale_x, img.shape[0]*scale_y
    left_x, right_x = center_x+width_scaled/2.1 , center_x+width_scaled
    top_y, bottom_y = center_y - width_scaled/1.5, center_y
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped


def contour_det(in_path,o_path, choise):
    img = cv2.imread(in_path, 1)
    detector = dlib.get_frontal_face_detector()
    #win = dlib.image_window()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    can = cv2.Canny(gray,20, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(can, cv2.MORPH_CLOSE, kernel)

    _, contours0, hierarchy = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    a_max = 0
    c = 0
    for cnt in contours0: #Определение всех контуров
        perimeter = cv2.arcLength(cnt,True)
        c+=1
        if perimeter>a_max:
            x,y,w,h = cv2.boundingRect(cnt)
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            a_max = perimeter
            rect = cv2.minAreaRect(cnt)
            box= cv2.boxPoints(rect)
            box = np.int0(box)

    center = (int(rect[0][0]),int(rect[0][1]))
    area = int(rect[1][0]*rect[1][1])

    edge1 = np.int0((box[1][0] - box[0][0],box[1][1] - box[0][1]))
    edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))

    usedEdge = edge1
    if cv2.norm(edge2) > cv2.norm(edge1):
        usedEdge = edge2

    reference = (1,0) # горизонтальный вектор, задающий горизонт

    # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
    angle = 180.0/math.pi * math.acos((reference[0]*usedEdge[0] + reference[1]*usedEdge[1]) / (cv2.norm(reference) *cv2.norm(usedEdge)))

    if area >500:
        cv2.drawContours(img,[box],0,(255,0,0),2) # рисуем прямоугольник

    new_img = img[y:y+h, x: x+w] #Создаем новое изображение из самого большого контура
    #print(h,w)
    # Выравниваем изображение
    if w>h:
        new_img =imutils.rotate_bound(new_img, angle+90)# Если изображение горизонтальное, то поворачиваем до вертикального состояния
    else:
        new_img =imutils.rotate_bound(new_img, angle-90)

    #Промежуточное изображение для определения лица на паспорте (подумать над избмежанием такого решения)
    cv2.imwrite(os.path.join(o_path, '123.png'), new_img)
    img_1 = dlib.load_rgb_image(os.path.join(o_path, "123.png"))
    dets = detector(img_1, 1)
    #os.path.join(o_path, os.system("rm 123.png"))
    #Проверка на детекцию лица: если количество лиц = 0 то переворачиваем на 180
    if len(dets)==0:
        new_img =imutils.rotate_bound(new_img, 180)
    else:
        print("more than 0 faces")

    #Выбор метода сегментации изображения
    if choise == 0: #Вырезать с исходного изображения паспорт целиком
        #cv2.imwrite(os.path.join(o_path, str(c) + '.jpg'), new_img)
        num = crop_num(new_img, 0.85, 0.3)
        num = imutils.rotate_bound(num, -90)
        cv2.imwrite(str(o_path) + "num.jpg", num)
        return new_img, num
    elif choise == 1: #Вырезать 2 страницы паспорта
        img,(rh,rw) = resize_image(new_img)
        first_page = crop_first_page(img, 0.85, 0.3)
        second_page = crop_second_page(img, 0.85, 0.3)
        num = crop_num(img, 0.85, 0.3)
        num = imutils.rotate_bound(num, -90)
        cv2.imwrite(str(o_path) + "num.jpg", num)
        cv2.imwrite(str(o_path) + "1.jpg", first_page)
        cv2.imwrite(str(o_path) + "2.jpg", second_page)
        return first_page, second_page, num
    else:
        print("OOPSS")


def text_recogn(image, o_path):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    noisy_image = img_as_ubyte(gray)

    bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(30), s0=20, s1=5)

    txt = open(os.path.join(o_path, "text_recog.txt"), "w")
    text = image_to_string(bilat, lang='rus')
    text = corrector.FixFragment(re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
    txt.write(text)
    txt.close

    #txt_name_list = get_text(input_path)
    #for name in txt_name_list: # Редактирование распознанного текста
    txt = open(os.path.join(output_path, "red_text_recog.txt"), "w")
    with open(os.path.join(output_path, "text_recog.txt")) as f:
        f = f.read().splitlines()
        #print(f)
        for line in f:
            for word in line.split(' '):
                if word.isupper() or re.search('\d\d\.\d\d\.\d{4}', word) or re.search('\№\d*', word):
                    word = corrector.FixFragment(word)
                    txt.write(word+' ')
            txt.write('\n')
    txt.close
    #os.path.join(o_path, os.system("rm text_recog.txt"))

def text_recogn_f_page(f_page, o_path):

    gray = cv2.cvtColor(f_page, cv2.COLOR_BGR2GRAY)

    noisy_image = img_as_ubyte(gray)

    bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(30), s0=20, s1=5)

    txt = open(os.path.join(o_path, "text_recog_1.txt"), "w")
    text = image_to_string(bilat, lang='rus')
    text = corrector.FixFragment(re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
    txt.write(text)
    txt.close

    #txt_name_list = get_text(input_path)
    #for name in txt_name_list: # Редактирование распознанного текста
    txt = open(os.path.join(output_path, "red_text_recog_1.txt"), "w")
    with open(os.path.join(output_path, "text_recog_1.txt")) as f:
        f = f.read().splitlines()
        #print(f)
        for line in f:
            for word in line.split(' '):
                if word.isupper() or re.search('\d\d\.\d\d\.\d{4}', word) or re.search('\№\d*', word):
                    word = corrector.FixFragment(word)
                    txt.write(word+' ')
            txt.write('\n')
    txt.close
    #os.path.join(o_path, os.system("rm text_recog_1.txt"))

def text_recogn_sec_page(sec_page, o_path):
    gray = cv2.cvtColor(sec_page, cv2.COLOR_BGR2GRAY)

    noisy_image = img_as_ubyte(gray)

    bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(30), s0=20, s1=5)

    txt = open(os.path.join(o_path, "text_recog_2.txt"), "w")
    text = image_to_string(bilat, lang='rus')
    text = corrector.FixFragment(re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
    txt.write(text)
    txt.close

    #txt_name_list = get_text(input_path)
    #for name in txt_name_list: # Редактирование распознанного текста
    txt = open(os.path.join(output_path, "red_text_recog_2.txt"), "w")
    with open(os.path.join(output_path, "text_recog_2.txt")) as f:
        f = f.read().splitlines()
        #print(f)
        for line in f:
            for word in line.split(' '):
                if word.isupper() or re.search('\d\d\.\d\d\.\d{4}', word) or re.search('\№\d*', word):
                    word = corrector.FixFragment(word)
                    txt.write(word+' ')
            txt.write('\n')
    txt.close
    #os.path.join(o_path, os.system("rm text_recog_2.txt"))

def num_recogn_page(num_page, o_path):
    gray = cv2.cvtColor(num_page, cv2.COLOR_BGR2GRAY)

    noisy_image = img_as_ubyte(gray)

    bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(30), s0=20, s1=5)

    txt = open(os.path.join(o_path, "num_recog.txt"), "w")
    text = image_to_string(bilat, lang='rus')
    text = corrector.FixFragment(re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
    txt.write(text)
    txt.close


corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('ru_small.bin')

####################################################
#Операция выделения требуемого изображения из исходного
#Здесь был код из contour_det()

choise = int(input("0 or 1: "))
if choise == 1:
    first_p, sec_p, num = contour_det(input_path, output_path, choise)
    text_recogn_f_page(first_p, output_path)
    text_recogn_sec_page(sec_p, output_path)
    num_recogn_page(num, output_path)
elif choise == 0:
    new_img, num = contour_det(input_path, output_path, choise)
    text_recogn(new_img, output_path)
    num_recogn_page(num, output_path)

####################################################
