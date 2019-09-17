import cv2
from pytesseract import image_to_string
import numpy as np
import os
import jamspell
import re
from skimage import img_as_ubyte
from skimage.filters.rank import mean_bilateral
from skimage.morphology import disk
import imutils
import gui as gui

# input_path = r'/home/paul/Desktop/Work_projects/images/1_page.jpg'
input_path = r'/home/paul/Desktop/Work_projects/images/img.png'
output_path = r'/home/paul/Desktop/Work_projects/result'

sex = ['МУЖ', 'ЖЕН']
exc_words = ['ПО', 'В', 'НА']


def ReadFromFile(input):
	mas_1 = []
	mas_2 = []
	txt = open(os.path.join(input, "red_text_recog_1.txt"), "r")
	f = txt.read().splitlines()
	for line in f:
		for word in line.split(' '):
			if (word not in exc_words and len(word) <= 2 or (
					word.isspace() or word == '')):
				continue
			else:
				mas_1.append(word)
	txt.close()

	txt = open(os.path.join(input, "red_text_recog_2.txt"), "r")
	f = txt.read().splitlines()
	for line in f:
		for word in line.split(' '):
			if (word not in exc_words and len(word) == 2 or (
					word.isspace() or word == '') or word == '№'):
				continue
			else:
				mas_2.append(word)
	txt.close()

	return mas_1, mas_2


def PlaceWhereTaken(input_mas_1):
	place = []
	for word in input_mas_1:
		if re.search('\d\d\.\d\d\.\d{4}', word):
			continue
		else:
			place.append(word)
	place = ' '.join(place)
	if place == '':
		place = None
		return place
	else:
		return place


def DataOfTaken(input_mas_1):
	for word in input_mas_1:
		if re.search('\d\d\s{0,3}\.\d\d\s{0,3}\.\d{4}', word):
			data_taken = word
			input_mas_1.remove(data_taken)
			return data_taken
		else:
			continue
	return None


def Key(input_mas_1):
	for word in input_mas_1:
		if re.search('\d\d\d\-\d\d\d', word):
			key = word
			input_mas_1.remove(key)
			return key
		else:
			continue
	return None


def FIO(input_mas):
	fio = []
	c = 0

	for i in range(len(input_mas)):
		if (re.search('\d\d\.\d\d\.\d{4}', input_mas[c])) or (input_mas[c] in sex):
			break
		else:
			fio.append(input_mas[c])
			input_mas.remove(input_mas[c])

	if len(fio) != 3:
		fio = None
		return fio
	else:
		fio = ' '.join(fio)
		return [fio]


def Sex(input_mas):
	for word in input_mas:
		if sex[0] in word:
			# s1 = input_mas.index(word)
			s = sex[0]
			input_mas.remove(word)
			# print(input_mas)
			return s
		elif sex[1] in word:
			# s1 = input_mas.index(word)
			s = sex[1]
			input_mas.remove(word)
			# print(input_mas)
			return s
		else:
			continue


	return None


def Birthday(input_mas):
	for word in input_mas:
		if re.search('\d\d\.\d\d\.\d{4}', word):
			birthday = word
			# b = input_mas.index(word)
			input_mas.remove(word)
			return birthday
		else:
			continue


# return birthday


def BirthPlace(input_mas_2):
	# mas = []
	if input_mas_2 == '':
		return None
	else:
		input_mas_2 = ' '.join(input_mas_2)
		return input_mas_2


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


def crop_first_page(img, scale_x=1.0, scale_y=1.0):
	center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
	width_scaled, height_scaled = img.shape[1] * scale_x, img.shape[0] * scale_y

	left_x, right_x = center_x - width_scaled*2.3, center_x + width_scaled / 2
	top_y, bottom_y = center_y - height_scaled * 1.5, center_y - height_scaled / 1.5

	img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
	cv2.imwrite(os.path.join(output_path, '1_page.jpg'), img_cropped)
	return img_cropped


def crop_second_page(img, scale_x=1.0, scale_y=1.0):
	center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
	width_scaled, height_scaled = img.shape[1] * scale_x, img.shape[0] * scale_y

	left_x, right_x = center_x - width_scaled * 1.4, center_x + width_scaled / 2
	top_y, bottom_y = center_y, center_y + height_scaled * 0.85

	img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
	cv2.imwrite(os.path.join(output_path, '2_page.jpg'), img_cropped)
	return img_cropped


def crop_num(img, scale_x=1.0, scale_y=1.0):
	center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
	width_scaled, height_scaled = img.shape[1] * scale_x, img.shape[0] * scale_y

	left_x, right_x = center_x + width_scaled / 2.1, center_x + width_scaled
	top_y, bottom_y = center_y - width_scaled / 1.5, center_y

	img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
	return img_cropped


def crop_face(img, scale_x=1.0, scale_y=1.0):
	center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
	width_scaled, height_scaled = img.shape[1] * scale_x, img.shape[0] * scale_y

	left_x, right_x = center_x - width_scaled / 2, center_x - height_scaled * 0.4
	top_y, bottom_y = center_y + width_scaled * 0.15, center_y + height_scaled

	img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
	return img_cropped


def contour_det(in_path, o_path):  # , choise
	img = cv2.imread(in_path, 1)

	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# gray = cv2.bilateralFilter(gray, 11, 17, 17)
	# can = cv2.Canny(gray, 20, 150)
	#
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
	# closed = cv2.morphologyEx(can, cv2.MORPH_CLOSE, kernel)

	# _, contours0, hierarchy = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#
	# img_1 = img.copy()
	#
	# a_max = 0
	# for cnt in contours0: # Определение всех контуров
	#     perimeter = cv2.arcLength(cnt, True)
	#     if perimeter>a_max:
	#         x, y, w, h = cv2.boundingRect(cnt)
	#         img_1 = cv2.rectangle(img_1, (x, y), (x+w, y+h), (0, 255, 0), 2)
	#         a_max = perimeter
	#         rect = cv2.minAreaRect(cnt)
	#         box= cv2.boxPoints(rect)
	#         box = np.int0(box)
	#
	# area = int(rect[1][0]*rect[1][1])
	#
	# edge1 = np.int0((box[1][0] - box[0][0], box[1][1] - box[0][1]))
	# edge2 = np.int0((box[2][0] - box[1][0], box[2][1] - box[1][1]))
	#
	# usedEdge = edge1
	# if cv2.norm(edge2) > cv2.norm(edge1):
	#     usedEdge = edge2
	#
	# reference = (1,0) # горизонтальный вектор, задающий горизонт
	#
	# # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
	# angle = 180.0/math.pi * math.acos((reference[0]*usedEdge[0] + reference[1]*usedEdge[1]) / (cv2.norm(reference) *cv2.norm(usedEdge)))
	#
	# cv2.imwrite(os.path.join(o_path, 'contours.jpg'), img_1)
	#
	# if area >700:
	#     #cv2.drawContours(img,[box],0,(255,0,0),2) # рисуем прямоугольник
	#     mask = np.zeros_like(img)
	#     cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
	#     new_img = np.zeros_like(img)
	#     new_img[mask == (255, 255, 255)] = img[mask == (255, 255, 255)]
	#     cv2.imwrite(os.path.join(o_path, 'mask.png'), new_img)
	#     # Now crop
	#     x = np.where(mask == (255, 255, 255))[0]
	#     y = np.where(mask == (255, 255, 255))[1]
	#     (topx, topy) = (np.min(x), np.min(y))
	#     (bottomx, bottomy) = (np.max(x), np.max(y))
	#     new_img = new_img[topx:bottomx, topy:bottomy]
	# #
	# # Выравниваем изображение
	new_img = imutils.rotate_bound(img,
								   90)  # Если изображение горизонтальное, то поворачиваем до вертикального состояния

	# Промежуточное изображение для определения лица на паспорте (подумать над избмежанием такого решения)
	cv2.imwrite(os.path.join(o_path, '123.png'), new_img)
	cv2.imwrite(os.path.join(o_path, 'final.png'), new_img)

	img, (rh, rw) = resize_image(new_img)

	return img


def text_recogn(image, o_path):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	gray, (x, y) = resize_image(gray)
	noisy_image = img_as_ubyte(gray)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	cl1 = clahe.apply(noisy_image)

	bilat = mean_bilateral(cl1.astype(np.uint16), disk(30), s0=20, s1=5)

	txt = open(os.path.join(o_path, "text_recog.txt"), "w")
	text = image_to_string(bilat, lang='rus')
	text = corrector.FixFragment(
		re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
	txt.write(text)
	txt.close

	# txt_name_list = get_text(input_path)
	# for name in txt_name_list: # Редактирование распознанного текста
	txt = open(os.path.join(output_path, "red_text_recog.txt"), "w")
	with open(os.path.join(output_path, "text_recog.txt")) as f:
		f = f.read().splitlines()
		# print(f)
		for line in f:
			for word in line.split(' '):
				if (word.isupper() or re.search('\d\d\.\d\d\.\d{4}',
												word) or re.search('\№\d*',
																   word)):
					word = corrector.FixFragment(word)
					txt.write(word + ' ')
			txt.write('\n')
	txt.close


# os.path.join(o_path, os.system("rm text_recog.txt"))


def text_recogn_f_page(f_page, o_path):
	gray = cv2.cvtColor(f_page, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	gray, (x, y) = resize_image(gray)
	noisy_image = img_as_ubyte(gray)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	cl1 = clahe.apply(noisy_image)

	bilat = mean_bilateral(cl1.astype(np.uint16), disk(30), s0=20, s1=5)

	txt = open(os.path.join(o_path, "text_recog_1.txt"), "w")
	text = image_to_string(bilat, lang='rus')
	text = corrector.FixFragment(
		re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%|\[|\]|\,', '', text))
	# text = str(text.split())
	txt.write(text)
	txt.close

	# txt_name_list = get_text(input_path)
	# for name in txt_name_list: # Редактирование распознанного текста
	txt = open(os.path.join(output_path, "red_text_recog_1.txt"), "w")
	with open(os.path.join(output_path, "text_recog_1.txt")) as f:
		f = f.read().splitlines()
		for line in f:
			for word in line.split(' '):
				if word.isupper() or re.search('\d\d\.\d\d\.\d{4}',
											   word) or re.search('\№\d*',
																  word) or re.search(
					'\d\d\d\-\d\d\d', word):
					word = corrector.FixFragment(word)
					txt.write(word + ' ')
			txt.write('\n')
	txt.close


# os.path.join(o_path, os.system("rm text_recog_1.txt"))


def text_recogn_sec_page(sec_page, o_path):
	gray = cv2.cvtColor(sec_page, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	gray, (x, y) = resize_image(gray)

	noisy_image = img_as_ubyte(gray)
	bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(30), s0=15, s1=5)

	txt = open(os.path.join(o_path, "text_recog_2.txt"), "w")
	text = image_to_string(bilat, lang='rus')
	text = corrector.FixFragment(
		re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
	txt.write(text)
	txt.close

	# txt_name_list = get_text(input_path)
	# for name in txt_name_list: # Редактирование распознанного текста
	txt = open(os.path.join(output_path, "red_text_recog_2.txt"), "w")
	with open(os.path.join(output_path, "text_recog_2.txt")) as f:
		f = f.read().splitlines()
		# print(f)
		for line in f:
			for word in line.split(' '):
				if (word.isupper() and len(word) > 2) or re.search(
						'\d\d\.\d\d\.\d{4}', word) or re.search('\№\d*', word):
					word = corrector.FixFragment(word)
					txt.write(word + ' ')
			txt.write('\n')
	txt.close


def num_recogn_page(num_page, o_path):
	gray = cv2.cvtColor(num_page, cv2.COLOR_BGR2GRAY)

	# noisy_image = img_as_ubyte(gray)

	# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	# cl1 = clahe.apply(gray)

	bilat = mean_bilateral(gray.astype(np.uint16), disk(30), s0=20, s1=5)

	txt = open(os.path.join(o_path, "num_recog.txt"), "w")
	text = image_to_string(bilat, lang='rus')
	text = corrector.FixFragment(
		re.sub(r'\?|\!|\/|\;|\:|\=|\_|\(|\[|\)|\]|\#|\,|\%', '', text))
	txt.write(text)
	txt.close


corrector = jamspell.TSpellCorrector()
corrector.LoadLangModel('ru_small.bin')

person_data = {'taken place': None, 'taken data': None, 'key': None,
			   'FIO': None, 'sex': None, 'birthday': None,
			   'place': None}
person_data_func = {'taken place': PlaceWhereTaken, 'taken data': DataOfTaken,
					'key': Key, 'FIO': FIO, 'sex': Sex, 'birthday': Birthday,
					'place': BirthPlace}
FIO_dict = [None, None, None]

####################################################

stream = cv2.VideoCapture(1)
stream.set(3, 1280)
stream.set(4, 960)
stream.set(14, 10)

cv2.namedWindow('i', cv2.WINDOW_NORMAL)

RUN = True

while RUN:
	ret, img = stream.read()
	if not ret:
		RUN = False
	else:
		cv2.imshow('i', img)
		if cv2.waitKey(10) & 0xFF == 27:
			cv2.imwrite(
				'/home/paul/Desktop/Work_projects/images/img.png', img)
			RUN = False

cv2.destroyAllWindows()
stream.release()

img = contour_det(input_path, output_path)
first_page = crop_first_page(img, 0.85, 0.3)
second_page = crop_second_page(img, 0.85, 0.3)
# num = crop_num(img, 0.85, 0.3)
# face = crop_face(img, 0.85, 0.3)

# cv2.imwrite(os.path.join(output_path, 'num.jpg'), num)
# cv2.imwrite(os.path.join(output_path, 'face.jpg'), face)

text_recogn_f_page(first_page, output_path)
text_recogn_sec_page(second_page, output_path)
# num_recogn_page(num, output_path)

mas_1, mas_2 = ReadFromFile(output_path)

# for key in person_data.keys():
# 	if (key == 'taken data' or key == 'taken place' or key == 'key'):
# 		person_data[key] = person_data_func[key](mas_1)
# 	elif (key == 'sex' or key == 'FIO' or key == 'birthday' or key == 'place'):
# 		person_data[key] = person_data_func[key](mas_2)
# 	else:
# 		continue
person_data['taken data'] = person_data_func['taken data'](mas_1)
person_data['key'] = person_data_func['key'](mas_1)
person_data['taken place'] = person_data_func['taken place'](mas_1)

person_data['sex'] = person_data_func['sex'](mas_2)
person_data['FIO'] = person_data_func['FIO'](mas_2)
person_data['birthday'] = person_data_func['birthday'](mas_2)
person_data['place'] = person_data_func['place'](mas_2)

print(person_data)

while not all(person_data.values()):
	stream = cv2.VideoCapture(1)
	stream.set(3, 1280)
	stream.set(4, 960)
	stream.set(14, 10)

	cv2.namedWindow('i', cv2.WINDOW_NORMAL)

	RUN = True

	while RUN:
		ret, img = stream.read()
		if not ret:
			RUN = False
		else:
			cv2.imshow('i', img)
			if cv2.waitKey(10) & 0xFF == 27:
				cv2.imwrite(
					'/home/paul/Desktop/Work_projects/images/img.png', img)
				RUN = False

	cv2.destroyAllWindows()
	stream.release()

	img = contour_det(input_path, output_path)
	first_page = crop_first_page(img, 0.85, 0.3)
	second_page = crop_second_page(img, 0.85, 0.3)

	text_recogn_f_page(first_page, output_path)
	text_recogn_sec_page(second_page, output_path)

	mas_1, mas_2 = ReadFromFile(output_path)

	for key in person_data.keys():
		if person_data[key] is None:
			if key == 'taken data' or key == 'taken place'	or key == 'key':
				person_data[key] = person_data_func[key](mas_1)
			elif key == 'sex' or key == 'FIO' or key == 'birthday' or key == 'place':
				person_data[key] = person_data_func[key](mas_2)
		else:
			continue

	# if key == 'FIO':
	#     fio = []
	#     fio = person_data_func[key]()
	#     for i in range(3):
	#         try:
	#             FIO_dict[i] = fio[i]
	#         except IndexError:
	#             continue
	#     person_data[key] = FIO_dict

	print(person_data)
gui.run_gui_passport(person_data)