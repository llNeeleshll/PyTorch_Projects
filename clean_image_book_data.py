from os import listdir
from PIL import Image
import os

folder_location = 'C:/Users/neele/OneDrive/Documents/Datasets/booksimages/booksimages/'

total_images = len(listdir(folder_location))
good_images = 0
deleted_images = 0

for filename in listdir(folder_location):    
    try:
        img = Image.open(folder_location + filename) # open the image file
        img.verify() # verify that it is, in fact an image
        good_images += 1
    except (IOError, SyntaxError) as e:        
        os.remove(folder_location + filename)
        deleted_images += 1
        
    verified_images = good_images + deleted_images
    print("Verified {0} images out of {1} ==> {2} are good and {3} are deleted".format(verified_images, total_images, good_images, deleted_images))