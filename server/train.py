import cv2, os, shutil, pywt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')
path_to_data = "./dataset/"
path_to_cr_data = "./dataset/cropped/"

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            return roi_color

def generate_cropped_imgs():
    # list img dirs
    img_dirs = []
    for entry in os.scandir(path_to_data):
        if entry.is_dir():
            img_dirs.append(entry.path)

    # delete crop path if already exists
    if os.path.exists(path_to_cr_data):
        shutil.rmtree(path_to_cr_data)
    os.mkdir(path_to_cr_data)

    cropped_image_dirs = []
    celebrity_file_names_dict = {}
    for img_dir in img_dirs:
        count = 1
        celebrity_name = img_dir.split('/')[-1]
        celebrity_file_names_dict[celebrity_name] = []
        for entry in os.scandir(img_dir):
            roi_color = get_cropped_image_if_2_eyes(entry.path)
            if roi_color is not None:
                cropped_folder = path_to_cr_data + celebrity_name
                if not os.path.exists(cropped_folder):
                    os.makedirs(cropped_folder)
                    cropped_image_dirs.append(cropped_folder)
                    # print("Generating cropped images in folder: ",cropped_folder)
                cropped_file_name = celebrity_name + str(count) + ".png"
                cropped_file_path = cropped_folder + "/" + cropped_file_name
                cv2.imwrite(cropped_file_path, roi_color)
                celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
                count += 1
    return cropped_image_dirs, celebrity_file_names_dict

def get_class_dicts(celebrity_file_names_dict):
    class_dict = {}
    count = 0
    for celebrity_name in celebrity_file_names_dict.keys():
        class_dict[celebrity_name] = count
        count = count + 1
    
    return class_dict


def prepare_x_y(celebrity_file_names_dict, class_dict):
    X, y = [], []
    for celebrity_name, training_files in celebrity_file_names_dict.items():
        for training_image in training_files:
            img = cv2.imread(training_image)
            scalled_raw_img = cv2.resize(img, (32, 32))
            img_har = w2d(img,'db1',5)
            scalled_img_har = cv2.resize(img_har, (32, 32))
            combined_img = np.vstack((scalled_raw_img.reshape(32*32*3,1),scalled_img_har.reshape(32*32,1)))
            X.append(combined_img)
            y.append(class_dict[celebrity_name])  
    return X, y

def train_random_forest(x, y):
    model = RandomForestClassifier()
    model.fit(x, y)
    return model

if __name__ == "__main__":
    cropped_image_dirs, celebrity_file_names_dict = generate_cropped_imgs()
    class_dict = get_class_dicts(celebrity_file_names_dict)
    x, y = prepare_x_y(celebrity_file_names_dict, class_dict)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
    model = train_random_forest(x_train, y_train)
    print(model.score(x_test, y_test))
