import numpy as np
from PrepareDataset import *
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
import pickle
import tkinter as tk
from tkinter import simpledialog,messagebox

#########################################################################################


called=0


def add_face():
    global called
    try:
        person_name=simpledialog.askstring("Name","Enter your name")
        cam = cv2.VideoCapture(0)
    
        folder = "people/" + person_name.lower()
        try:
            os.mkdir(folder)
    
            flag_start_capturing = False
            sample = 1
            cv2.namedWindow("Face", cv2.WINDOW_AUTOSIZE)
    
            while True:
                ret, frame = cam.read()
    
                faces_coord = detect_face(frame) 
    
                if len(faces_coord):
                    faces = normalize_faces(frame, faces_coord)
                    cv2.imwrite(folder + '/' + str(sample) + '.jpg', faces[0])  
    
                    if flag_start_capturing:
                        sample += 1
    
                draw_rectangle(frame, faces_coord)  
                cv2.imshow('Face', frame)  
                keypress = cv2.waitKey(1)   
    
                if keypress == ord('c'):   
    
                    if not flag_start_capturing:    # Not False means True.
                        flag_start_capturing = True
    
                if sample > 150:   
                    break
    
            cam.release()          
            cv2.destroyAllWindows()   
            


            if called==0:
                global name_entry
                #name_frame=tk.Frame(root).pack()
                name_label=tk.Label(root,text="New face added ",font=('Times',12,'bold'),fg='blue').pack(side=tk.LEFT,padx=20)
                name_entry=tk.StringVar()
                names_entry=tk.Entry(root,textvariable=name_entry,font=('Times',12,'bold'),state='readonly').pack(side=tk.LEFT,padx=20)
                name_entry.set(person_name)
            else:
                    name_entry.set(person_name)
            
            
        except :
            messagebox.askokcancel("Already present","Data already available")
        called=called+1
        
    except AttributeError:
        messagebox.askokcancel("Name error","Name is required")
        
        
####################################################################################################################################################################################################################################################################


def x():
    images = []
    labels = []
    labels_dic = {}
         
    ###############     function collect_dataset ###############################################
    
    def collect_dataset():
    
        people = [person for person in os.listdir("people/")]
    
        for i, person in enumerate(people):
            labels_dic[i] = person
            for image in os.listdir("people/" + person):
                if image.endswith('.jpg'):
                    images.append(cv2.imread("people/" + person + '/' + image, 0))
                    labels.append(i)
        return images, np.array(labels), labels_dic
    
    
    ########################### function collect_dataset end ###################################
    
    images, labels, labels_dic = collect_dataset()
    
    X_train = np.asarray(images)    #np.asarray()  used to convert the input into ndarray. if alrady input is an array then then it return the same input.
    train = X_train.reshape(len(X_train), -1)
    
    sc = StandardScaler()           # standardization is all about scaling your data  in such a way  that  all the variables  and
                                    #     their values lie within similar range. z=(variable value - mean)/std. deviation
    X_train_sc = sc.fit_transform(train.astype(np.float64))
    
    pca1 = PCA(n_components=.97)        # PCA - principal component analysis. dimension reduction process (using eigenvector and    values)
    new_train = pca1.fit_transform(X_train_sc)
    kf = KFold(n_splits=5,shuffle=True)     # split the data in k- consecutive fold where each fold is used for the test
    
    param_grid = {'C': [.0001, .001, .01, .1, 1, 10]}
    
    gs_svc = GridSearchCV(SVC(kernel='linear', probability=True), param_grid=param_grid, cv=kf, scoring='accuracy')
    gs_svc.fit(new_train, labels)
    clf = gs_svc.best_estimator_
    filename = 'svc_linear_face.pkl'
    f = open(filename, 'wb')
    pickle.dump(clf, f)
    f.close()
    
    filename = 'svc_linear_face.pkl'
    svc1 = pickle.load(open(filename, 'rb'))
    
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.namedWindow("opencv_face ", cv2.WINDOW_AUTOSIZE)
    
    while True:
        ret, frame = cam.read()
    
        faces_coord = detect_face(frame)  # detect more than one face
        if len(faces_coord):
            faces = normalize_faces(frame, faces_coord)
    
            for i, face in enumerate(faces):  # for each detected face
    
                t = face.reshape(1, -1)
                t = sc.transform(t.astype(np.float64))
                test = pca1.transform(t)
                prob = svc1.predict_proba(test)
                confidence = svc1.decision_function(test)
    
                pred = svc1.predict(test)
                #print(pred, pred[0])
    
                name = labels_dic[pred[0]].capitalize()
                #print(name)
    
                cv2.putText(frame, name, (faces_coord[i][0], faces_coord[i][1] - 10),
                            cv2.FONT_HERSHEY_PLAIN, 2, (66, 53, 243), 2)
    
            draw_rectangle(frame, faces_coord)  # rectangle around face
    
        cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                    cv2.LINE_AA)
    
        cv2.imshow("opencv_face", frame)  # live feed in external
        if cv2.waitKey(5) == 27:
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
  ######################################################################################################################################################  
    

root=tk.Tk()
root.title("GUI_for_PROJECT")


heading=tk.Label(root,text="WELCOME TO FACE RECOGNITION",fg='green',font=('Times',20,'bold')).pack()

capture_new_face=tk.Button(root,text="  Add New Face  ",bd=4,fg="red",font=('Times',15,'bold'),command=add_face).pack(pady=6)

detect_old_face=tk.Button(root,text="    Detect Face   ",bd=4,fg='red',font=('Times',15,'bold'),command=x).pack()

root.resizable(0,0)
root.mainloop()

