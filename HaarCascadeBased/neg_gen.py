import cv2
import os


def create_neg():
    neg_img = "/home/priya/Documents/pj2/haarcascade-negatives-master/images"
    if not os.path.exists('negative'):
    	os.mkdir('negative')
    num_pic=0001

    for i in os.listdir(neg_img):
    	print(i)
    	im_path = os.path.join(neg_img,i)
    	print(str(im_path))
    	if os.path.exists(im_path):
    		try:
    			img = cv2.imread(i)
    			#resize_img = cv2.resize(img,(100,100))
    			cv2.imwrite("negative/"+str(num_pic)+".jpg",img)
    			num_pic+=1
    		except Exception as e:
    			print(e)


def create_pos_n_neg():
    for file_type in ['neg']:
        
        for img in os.listdir(file_type):

            if file_type == 'pos':
                line = file_type+'/'+img+' 1 0 0 50 50\n'
                with open('info.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

create_pos_n_neg()