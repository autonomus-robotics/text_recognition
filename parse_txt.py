import os
import re

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
			if (word not in exc_words and len(word) <= 2 or (word.isspace() or word == '')):
				continue
			else:
				mas_1.append(word)
	txt.close()

	txt = open(os.path.join(input, "red_text_recog_2.txt"), "r")
	f = txt.read().splitlines()
	for line in f:
		for word in line.split(' '):
			if (word not in exc_words and len(word) == 2 or (word.isspace() or word == '') or word == '№'):
				continue
			else:
				mas_2.append(word)
	txt.close()

	return mas_1, mas_2


def place_where_taken(input_mas_1):
	place = []
	for word in input_mas_1:
		if re.search('\d\d\.\d\d\.\d{4}', word):
			continue
		else:
			place.append(word)
	place = ' '.join(place)
	return place

def data_of_taken(input_mas_1):
	for word in input_mas_1:
		if re.search('\d\d\.\d\d\.\d{4}', word):
			data = word
			return data
		else:
			data = None
	return data


def FIO(input_mas):
    s, s1 = Sex(input_mas)
    b, b1= Birthday(input_mas)
    if s1 == 3:
        surname, name, lastname = input_mas[0], input_mas[1], input_mas[2]
        input_mas.remove(surname)
        input_mas.remove(name)
        input_mas.remove(lastname)
    elif s1 == 2:
        surname, name, lastname = None, input_mas[0], input_mas[1]
        input_mas.remove(name)
        input_mas.remove(lastname)
    elif b1 == 3:
        surname, name, lastname = input_mas[0], input_mas[1], input_mas[2]
        input_mas.remove(surname)
        input_mas.remove(name)
        input_mas.remove(lastname)
    elif b1 == 2:
        surname, name, lastname = None, input_mas[0], input_mas[1]
        input_mas.remove(name)
        input_mas.remove(lastname)
    else:
        surname, name, lastname = None, None, None

    return surname, name, lastname, s, b

def Sex(input_mas):
    for word in input_mas:
        if sex[0] in word:
            s1 = input_mas.index(word)
            s = sex[0]
            input_mas.remove(word)
            #print(input_mas)
            return s, s1
        elif sex[1] in word:
            s1 = input_mas.index(word)
            s = sex[1]
            input_mas.remove(word)
            #print(input_mas)
            return s, s1
        else:
            s, s1 = None, None
    return s, s1

def Birthday(input_mas):
    for word in input_mas:
        if re.search('\d\d\.\d\d\.\d{4}', word):
            birthday = word
            b = input_mas.index(word)
            input_mas.remove(word)
            return birthday,b
        else:
            birthday, b = None, None
    return birthday,b


def BirthPlace(input_mas_2):
	mas = ' '.join(input_mas_2)
	return mas


mas_1, mas_2 = ReadFromFile(output_path)

place = place_where_taken(mas_1)
data_pas = data_of_taken(mas_1)

surname, name, lastname, s, birth = FIO(mas_2)
# s = Sex(mas_2)
# birth = Birthday(mas_2)
bp = BirthPlace(mas_2)
print("place: " + str(place), "\n data pas: " + str(data_pas),
        "\n surname: " + str(surname), "\n name: " + str(name), "\n lastname: " + str(lastname),
        "\n sex: " + str(s), "\n birthday: " + str(birth), "\n birth place: " + str(bp))
