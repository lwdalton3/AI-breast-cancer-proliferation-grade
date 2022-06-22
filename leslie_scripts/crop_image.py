import cv2 
import os
from tqdm import tqdm

#===Inputs===#
testFolder = "test"
saveDir = "test_panels"
#===Inputs===#

print("Loading file paths")
allFiles = []
for dirname, _, filenames in os.walk(testFolder): #WRITE TEST FOLDER NAME HERE #We will us OS.Walk Function to get all file names inside this 4 folders
    for filename in filenames:
        allFiles.append(os.path.join(dirname, filename))

if os.path.isdir(saveDir) == False: #If directory do not exist, make one
    os.makedirs(saveDir)
    print("Creating:",saveDir)

def crop_image(i):
    print(allFiles[i])
    img = cv2.imread(allFiles[i])
    y, x, z = img.shape
    current_x=0
    current_y=0
    list_img = []
    while current_x+250 <=x and current_y+250 <=y  :
        img_crop = img[current_y:current_y+250,current_x:current_x+250]
        current_x += 250
        if current_x+250>x:
            current_x=0
            current_y+=250
        list_img.append(img_crop)

    for i in range(len(list_img)):
        # list_img[i] = cv2.resize(list_img[i],(150,150))
        cv2.imwrite(os.path.join(saveDir,"img_crop"+str(i)+".jpeg"),list_img[i])

for i in tqdm(range(len(allFiles))):
    crop_image(i)
