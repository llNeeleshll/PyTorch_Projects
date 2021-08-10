import pytesseract
from PIL import Image
from skimage.segmentation import clear_border
import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import imutils
import pandas as pd

path = "C:/Users/neele/OneDrive/Documents/Datasets/ML_assignment_IDfy-20201223T105952Z-001/ML_assignment_IDfy/"

df = pd.read_csv(path + "dataset.csv")

def build_tesseract_options(psm=7):
    # tell Tesseract to only OCR alphanumeric characters
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    # set the PSM mode
    options += " --psm {}".format(psm)
    # return the built options string
    return options

def get_image_text(img_path):
    
    img = cv2.imread(img_path, 0)
    img_canny = cv2.Canny(img, 30, 200) 
    cnts = cv2.findContours(img_canny.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    licensePlate = None
    
    roi = []
    
    text_list = []

    for i, c in enumerate(cnts):
        
        (x, y, w, h) = cv2.boundingRect(c)
        ar = float(w) * float(h)
            
        licensePlate = img[y:y + h, x:x + w]
        roi = cv2.threshold(licensePlate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        if ar > 20000 :
            options = build_tesseract_options(psm=7)
            text = pytesseract.image_to_string(roi, config=options)
            
            if text != None and len(text) >= 6 :
                text_list.append(text)
            
        
    return text_list

def execute_code():
    file_name = path + 'normal/crop_m1/I00019.png'
    print(file_name)
    #captch_ex(file_name)
            
    correct = 0
    text_captured = 0

    text_dict = dict()

    for i, item in enumerate(list(df.iloc[:,0])):

        print("For image : {0}".format(str(i)))
        
        if "h" in item:
            item = "hdr/" + item
        else:
            continue
            item = "normal/" + item
        
        image_path = path + item
        output_text_label = list(df.iloc[:,1])[i]
        
        text_list = get_image_text(image_path)

        for extracted_item in text_list:
            if output_text_label in extracted_item:
                correct += 1
        
        text_dict[i] = (image_path, output_text_label, text_list)

    end = 0

execute_code()
        
    