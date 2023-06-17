# -*- coding:utf-8 -*-
import ntpath
import os

from paddleocr import PaddleOCR, draw_ocr, PPStructure
import json


def startRecognition(image_url):
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    # to switch the language model in order.
    folder, file_name = os.path.split(image_url)
    ocr = PaddleOCR(
        det_model_dir='models/det_db_inference/',
        rec_model_dir='models/v3_cyrillic_mobile/inference/',
        use_angle_cls=False,
        rec_char_dict_path='/Users/a.drozdov/Documents/Diplom work/Paddle ocr/PaddleOCR-release-2.6/ppocr/utils/dict/cyrillic_dict.txt')
    # need to run only once to download and load model into memory
    img_path = image_url
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw result
    from PIL import Image

    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    print(json.dumps(txts, sort_keys=True, indent=4, ensure_ascii=False))
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/Library/Fonts/Arial Unicode.ttf')
    im_show = Image.fromarray(im_show)
    path = os.path.join(folder, file_name.split('.')[0] + "_" + "rec." + file_name.split('.')[1])
    im_show.save(path)
    return os.path.join(folder, file_name.split('.')[0] + "_" + "rec." + file_name.split('.')[1])
