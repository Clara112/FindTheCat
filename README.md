# FindTheCat

Requires the following librarires:
- time ```$ install numpy ```
- numpy
- h5py
- matplotlib
- scipy 

Python program which has around 70% accuracy at recognising cats

```
#######################
# TEST YOUR OWN IMAGE #
#######################

my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_image = my_image/255.
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" 
              + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
```
