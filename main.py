from PIL import Image
import time
import os
import math
import numpy as np
import yaml


def get_ethalons_data():
    ethalons_data = {}
    for eth in ethalons_info:
        ethalons_data[eth] = []
        for i in range(len(ethalons_info[eth][0]) // 4):
            for x in range(ethalons_info[eth][0][0 + 4*i], ethalons_info[eth][0][2 + 4*i]):
                for y in range(ethalons_info[eth][0][1 + 4*i], ethalons_info[eth][0][3 + 4*i]):
                    ethalons_data[eth].append([pixels[y * width + x], x, y])
    return ethalons_data


def get_pixels():
    pixel_channels = []
    for shot in range(len(os.listdir(screens_path))):
        screen = Image.open(screens_path + os.listdir(screens_path)[shot])
        data = screen.getdata()
        if shot == 0:
            pixel_channels = [list(elem) for elem in data]
            continue
        next_screen_pixels = [list(elem) for elem in data]
        for index in range(len(pixel_channels)):
            pixel_channels[index] += next_screen_pixels[index]
    return pixel_channels


def euclidean_mahalanobis_metrics(ethalon_examples, x_pix_coord, y_pix_coord):
    dist_lst = []
    for eth in range(len(ethalon_examples)):
        distance = math.hypot((ethalon_examples[eth][1] - ethalon_examples[eth][2]), (x_pix_coord - y_pix_coord))
        dist_lst.append(distance)

    x_matrix = pixels[y_pix_coord * width + x_pix_coord]
    m_matrix = ethalon_examples[dist_lst.index(min(dist_lst))][0]
    x_minus_m = [x - m for x, m in zip(x_matrix, m_matrix)]
    x_matrix = np.transpose(x_matrix)
    cov_matrix = np.cov(x_matrix, bias=False)
    se = np.linalg.inv(np.sum([cov_matrix, np.eye(1)], axis=0))
    em_distance = np.sqrt(np.dot(np.dot(x_minus_m, se[0][0]), np.transpose(x_minus_m)))
    return em_distance


if __name__ == '__main__':
    start_time = time.time()

    # папка со снимками
    screens_path = './madagascar_screens/'
    # пиксели
    width, height = (Image.open(screens_path + '1.jpg')).size

    result_image = Image.new("RGB", (width, height))

    # ethalons_info = {
    #     0: [(1215, 1220), (466, 470), (0, 195, 255), 'Water'],       # Вода - голубой
    #     1: [(359, 364), (91, 96), (255, 229, 180), 'Mountain'],      # Горы - светло-коричневый
    #     2: [(952, 955), (225, 228), (101, 67, 33), 'Ground'],        # Земля - коричневый
    #     3: [(495, 500), (120, 125), (23, 114, 69), 'Forest'],        # Лес - зелёный
    #     4: [(967, 971), (39, 42), (102, 204, 170), 'Field'],         # Поле - светло-зелёный
    #     5: [(701, 704), (46, 47), (255, 0, 13), 'City'],              # Город - ярко-красный
    #     6: [(361, 365), (9, 13), (23, 114, 69), 'Very dark Forest']  # Очень темный лес - такой же как и обычный лес
    # }

    # ethalons_info = {
    #     0: [(257, 263), (52, 60), (73, 93, 92), 'Ocean'],  # Океан
    #     1: [(775, 780), (168, 173), (68, 88, 87), 'Bay'],  # Залив
    #     2: [(998, 1002), (354, 358), (43, 66, 72), 'River'],  # Река
    #     3: [(333, 337), (519, 523), (198, 157, 129), 'Sand'],  # Песок
    #     4: [(567, 573), (444, 450), (93, 75, 76), 'Ground'],  # Земля
    #     5: [(1276, 1281), (368, 373), (43, 57, 65), 'Forest'],  # Лес
    #     6: [(1444, 1450), (218, 222), (65, 63, 70), 'Rare Forest'],  # Редкий лес
    # }

    pixels = get_pixels()
    count_pix = len(pixels)

    ethalons_info = yaml.load(open('config.yaml'), Loader=yaml.Loader)
    ethalons_data = get_ethalons_data()

    for i in range(count_pix):
        e_m_distances = []
        x_coord = i % width
        y_coord = i // width
        for elem in ethalons_data:
            e_m_distances.append(euclidean_mahalanobis_metrics(ethalons_data[elem], x_coord, y_coord))
        affiliation = e_m_distances.index(min(e_m_distances))
        result_image.putpixel((x_coord, y_coord), ethalons_info[affiliation][1])
        print('% done: {:.3f}'.format(i / count_pix * 100))
        if i % 5000 == 0:
            result_image.save('result2.jpg')
        if count_pix / (i + 1) == 2:
            result_image.show()

    result_image.save('result2.jpg')
    print('Execution time: {:.3f}s'.format(time.time() - start_time))
