# import cv2
# import numpy as np
import os
from docx import Document
from correct_skew import *
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]


def make_pdf(imagename):
    # Get a searchable PDF
    pdf = pytesseract.image_to_pdf_or_hocr(imagename, extension='pdf')
    with open('test.pdf', 'w+b') as f:
        f.write(pdf) # pdf type is bytes by default


def mkdir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)

def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    mkdir("Cropped")
    mkdir("Lines")
    img = correct_skew(img_for_box_extraction_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image
    cv2.imwrite("Lines/Image_bin.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//70
     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("Lines/verticle_lines.jpg",verticle_lines_img)
# Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("Lines/horizontal_lines.jpg",horizontal_lines_img)
# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("Lines/img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    contours.sort(key=lambda x:get_contour_precedence(x, img.shape[1]))
    idx = 0
    boundingBoxes = []
    data = []
    for c in contours:
        # print(c)
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w<img.shape[1]*0.9 and h<img.shape[0]*0.9):
            idx += 1
            boundingBoxes.append((x, y, w, h))
            new_img = img[y:y+h, x:x+w]
            datastring = pytesseract.image_to_string(new_img, lang='eng+chi_sim+rus',config='--psm 6')
            data.append((idx, datastring, (x, y, w, h)))
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
    return boundingBoxes, data

    # d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    # (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)


boundingBoxes, data = box_extraction("C:/Users/ww/Desktop/work/images_tesseract/2.jpg", "C:/Users/ww/Desktop/work/table_read/Cropped/")
make_pdf("C:/Users/ww/Desktop/work/images_tesseract/2.jpg")
document = Document()
table = document.add_table(rows=1, cols=3)
for item in data:
    cells = table.add_row().cells
    cells[0].text = str(item[0])
    cells[1].text = item[1]
    cells[2].text = str(item[2])

document.add_page_break()
document.save('demo.docx')