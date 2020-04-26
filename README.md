# Gate Detection


## Background


This algorithm is developed to detect gates for the AIRR racing drone. In the AIRR competition, the drone is asked to pass through gates which are set by the competition holder as fast as possible. The ultimate goal is to defeat the human pilot. It requires the computer vision algorithm to recognize and localize the gates accurately and efficiently. Due to the impressive performance of the deep learning based object detection algorithm, here I train a Fully Convolutional Network (FCN) to detect the gates pixel by pixel. Compare to other network trained for segmentation, FCN is more efficient by eliminating redundant convolutional computation. More infomation about the FCN can be found in this [paper][1].  


## How to use the code
### Environment
I worked in the [Google Colaboratory][2] to use its free GPU. You can download the ipynb files and upload them to your own google drive.
I used the MXNET frame. Check your cuda version using the following code and install the corresponding MXNET. 

```
!cat /usr/local/cuda/version.txt
```


### Data
I used the dataset of the competition, you can use other dataset. Save the dataset under the same folder with the ipynb files. The data should have the same shape to realize the batch training, I uniform the shape by discrading images with non-standard shapes in Gate_Detection.ipynb and cropping into two segments in Gate_Detection_Data_Augmentaion.ipynb. Other methods like random cropping should also work, but resizing the image will lead to problems.

### Networks
The FCN networks were build over a Resnet18. The last two layers were replaced by a convolutional layer and a transposed convolutional layer. 

```
net.add(nn.Conv2D(num_classes,kernel_size=1),
        nn.Conv2DTranspose(num_classes,kernel_size = 2s,padding = s/2,strides = s))
```
The transposed convolutional layer should be designed to convert the shape of the feature map to the original shape of the image. 


### Training

The *try_all_gpus* function will check how many gpus available and use all of them to train the networks. If no gpu is available, cpu will be used automatically.



  [1]: https://arxiv.org/abs/1411.4038
  [2]: https://colab.research.google.com
  
