from tkinter import *
from tkinter import filedialog
import numpy as np
from PIL import Image,ImageTk
import warnings
warnings.filterwarnings("ignore")
import my_package
from my_package import InstanceSegmentationModel
from my_package import package


File_Name=''
# Define the function you want to call when the filebrowser button is clicked.
def fileClick(clicked, segmentor):
    global File_Name
    global segm_image
    global bound_image
    global IMG
    f=filedialog.askopenfilename(initialdir=r"D:\pictures",filetypes=(("jpg files","*.jpg"),("all files","*.*")))
    if f=='': return
    File_Name = f
    IMG = ImageTk.PhotoImage(Image.open(File_Name))
    image = Image.open(File_Name)
    ar = np.array(image) / 255
    ar = ar.transpose(2, 0, 1)

    prediction_list = segmentor(ar)
    ml=prediction_list[1]
    Bl=prediction_list[0]
    ar = ar.transpose(1, 2, 0)
    br = ar.copy()
    height=len(ar)
    width=len(ar[0])
    colors = [np.array([0,200/255,0]),np.array([210/255,0,0]),np.array([0,0,200/255])]
    for i in range(3):
        mask=ml[i][0]
        bbox=Bl[i]
        package.apply_mask(ar,mask,colors[i],height,width)
        package.apply_bbox(br,bbox,height,width,colors[i])
    imge = Image.fromarray(np.uint8(ar * 255))
    imge.save(r"D:\pictures\a.jpg", 'JPEG')
    segm_image=ImageTk.PhotoImage(Image.open(r"D:\pictures\a.jpg"))
    imge = Image.fromarray(np.uint8(br * 255))
    imge.save(r"D:\pictures\b.jpg", 'JPEG')
    bound_image = ImageTk.PhotoImage(Image.open(r"D:\pictures\b.jpg"))
    image_label=Label(image=IMG).grid(row=1,column=0,columnspan=6)

    process(clicked)
    return






####### CODE REQUIRED (START) #######
# This function should pop-up a dialog for the user to select an input image file.
# Once the image is selected by the user, it should automatically get the corresponding outputs from the segmentor.
# Hint: Call the segmentor from here, then compute the output images from using the `plot_visualization` function and save it as an image.
# Once the output is computed it should be shown automatically based on choice the dropdown button is at.
# To have a better clarity, please check out the sample video.


####### CODE REQUIRED (END) #######

# `process` function definition starts from here.
# will process the output when clicked.
def process(clicked):
        if File_Name=='':
            e.delete(0,END)
            e.insert(0,"You have not selected any file!   Select a file")
            return
        if clicked.get() == 'Segmentation':
            e.delete(0,END)
            image_label = Label(image=segm_image)
            image_label.grid(row=1, column=7)
        else :
            e.delete(0,END)
            image_label = Label(image=bound_image)
            image_label.grid(row=1, column=7)


####### CODE REQUIRED (START) #######
# Should show the corresponding segmentation or bounding boxes over the input image wrt the choice provided.
# Note: this function will just show the output, which should have been already computed in the `fileClick` function above.
# Note: also you should handle the case if the user clicks on the `Process` button without selecting any image file.

####### CODE REQUIRED (END) #######

# `main` function definition starts from here.

if __name__ == '__main__':
    ####### CODE REQUIRED (START) ####### (2 lines)
    # Instantiate the root window.
    root=Tk()
    # Provide a title to the root window.
    root.title("Image Viewer")
    ####### CODE REQUIRED (END) #######

    # Setting up the segmentor model.
    annotation_file = './data/annotations.jsonl'
    transforms = []

    # Instantiate the segmentor model.
    segmentor = InstanceSegmentationModel()
    # Instantiate the dataset.
   # dataset = Dataset(annotation_file, transforms=transforms)

    # Declare the options.
    options = ["Segmentation", "Bounding-box"]
    clicked = StringVar()
    clicked.set("Segmentation")

    e = Entry(root, width=70)
    e.grid(row=0, column=0)

    ####### CODE REQUIRED (START) #######
    # Declare the file browsing button
    brws=Button(root,text="Select a File",command=lambda:fileClick(clicked,segmentor)).grid(row=0,column=1)
    ####### CODE REQUIRED (END) #######

    ####### CODE REQUIRED (START) #######
    # Declare the drop-down button
    drop_menu=OptionMenu(root,clicked,*options).grid(row=0,column=2)
    ####### CODE REQUIRED (END) #######

    # This is a `Process` button, check out the sample video to know about its functionality
    process_Button = Button(root, text="Process", command=lambda:process(clicked))
    process_
    Button.grid(row=0, column=3)

####### CODE REQUIRED (START) ####### (1 line)
# Execute with mainloop()
    root.mainloop()
