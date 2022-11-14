<div align="center">
  <h2>CS-7641: Machine Learning - Group 43</h2><br>
</div>

<h3>Team Members:</h3> 
Jaiswal Shivanee, Sama Sai Srikesh Reddy, Anugundanahalli Ramachandra Reshma, Sivakumar Nikhil Viswanath, Khochare Janavi

<h2> Project Proposal Video</h2>
<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/zQJS_rvAnV0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

# Introduction and Background
Few products can last forever, more so food items. Expiration dates on products serve as an estimate for the healthy usage of the product. It is the date up to which food maintains its biological and physical stability. There are many cases where the information on the product label is hard to read and this is even more true in the case of a visually impaired person. With an increasing need to maintain the food quality standard, this remains an issue which needs to be tackled. 

For the scope of this project, we considered an image data set which has images of various fruits, vegetables, packed products and beverages, with expiry date in 13 different date formats for the packed food products. 

# Problem Definition and Motivation
Many grocery products are similar in shape and texture, making it difficult for visually impaired people to identify them using touch. Also, they cannot read expiry dates of products, necessary to ensure safe consumption. Thus, we aim to create a system providing audio feedback to such people by identifying the grocery product and the best-before/expiry date if mentioned.

<div align="center">
  	<img src="flow_chart_updated.png">
</div>

# Data Collection and Preprocessing
The data for the project was obtained from 2 different sources:
1. The dataset for image classification was obtained from "The Freiburg Groceries Dataset"[6] which is a publicly available dataset containing 5000 RGB images of various food classes. For the first part of our project we will classify the images broadly into 5 labels: Fruits, Vegetables, Beverages, Snacks and Other
2. The dataset for date detection was obtained from "ExpDate" dataset[5] which is again a publicly available dataset. This dataset has images corresponding to real images of products with their expiry dates, few images which have only dates from real products and few images which have synthetic dates of various formats which will be used in the date detection part of our project

For all the images in our dataset, we first started with an image compresssion with the help of Principal Component Analysis (PCA)[7][8]. The dataset has very high quality images (approximately 1000 x 1000 dimensions - since each image had a different size) which will increase our model's training time. So in order to reduce the training time, we used Principal Component Analysis (PCA) technique using 50 components which captured around 98% of the variation in the Blue channel, 97.5% of variation in the Red channel and around 98% of the variation in the Green channel. 

An example of image compressed with the help of PCA is as shown below:

<div align="center">
  	<img src="OI_RI.png">
</div>

To help tackle the problem of overfitting our model to the data-set in hand, we also tried to artificially expand our data-set by performing image augmentation on our original product images. We tried the following augmentation variations on our images:
1) Left-Right flip
2) Up-Down flip
3) 90° flip
4) 270° flip
5) Saturataion adjusted
6) Brightness adjusted
7) Gamma adjusted

An example of the image augmentation performed on one of the images is as follows. The original image is as follows:

<div align="center">
  	<img src="CAKE0000.png">
</div>

The augmented images for the above sample image as described in the order above are as shown below:
<p float="left">
  <img src="CAKE0000_Flip_LR_256x256.png" width="100"/>
  <img src="CAKE0000_Flip_UP_256x256.png" width="100"/>
  <img src="CAKE0000_ROT90_256x256.png" width="100"/>
  <img src="CAKE0000_ROT270_256x256.png" width="100"/>
  <img src="CAKE0000_SATURATE_256x256.png" width="100"/>
  <img src="CAKE0000_BRIGHT_256x256.png" width="100"/> 
  <img src="CAKE0000_GAMMA_256x256.png" width="100"/>
</p>

# Method

In our first part of the project, we have performed a image classification of the grocery images in our dataset. We split our original dataset of xxx images into a training set of xxx images and a test dataset of xxx images. As discussed earlier, we broadly classfied the images into 5 categories: Fruits, Vegetables, Snacks, Beverages, Dairy and Others.


<div align="center">
  	<img src="model-architecture.png"> <br>
    <em>ResNet50 model architecture used for Object Classification</em>
</div>


The Object Classification model classifies the products into seven different classes, Beverages, Fruits, Vegetables, Snacks, Dairy, Others and Packed (which is the ExpDate dataset) using a supervised learning approach. This approach [9] uses Transfer Learning where the weights of the classification neural network are obtained from a pretrained model which in this case is the ResNet50 model trained on the ImageNet [2] dataset. Then ResNet50 is used on top of the custom CNN layer for classification. The ResNet50 architecture includes 48 Convolution layers along with 1 MaxPool and 1 Average Pool layer. ResNet50 is used here because it makes it possible to train ultra deep neural networks. A network can contain hundreds or thousands of layers and still achieve great performance due to residual networks. ResNets have a deep residual learning framework containing shortcut connections that simply perform identity mappings. The advantage of such identity mappings is that without any additional parameters added to the model and without increasing computational time, the performance is improved. Compared to other ResNets, ResNet50 has a few changes, the shortcut connections previously skipped two layers but now they skip three layers and presence of 1 x 1 convolution layers in between. The learning rate was set to be 0.01. Batch learning was used with a batch size of 32. 

Once we have the image classified, the next step is to detect the probable bounding box of expiry date for a packed product. We will be using a supervised model for the localization of the bounding box. For this we plan to use images of the "ExpDate" dataset which consists of packed products along with the co-ordinates of bounding boxes of dates. After this, we will crop the image as per the coodinates of the bounding box and use Optical Character Recognition (OCR)[9] on this cropped image to detect the date.

For, the cropping and date extraction process using OCR, we started with converting the image to grayscale, then resized our image and applied morphological transforms to enhance the contrast of the pixels of the image. 

<div align="center">
  	<img src="ocr-image.jpg">
</div>

<!-- The aim of this project is to: 
* Identify the product using object classification (supervised learning) 
* Extract the expiry date information on the packages (unsupervised learning) 

This object classification task is an image classification task based on Transfer Learning. The base CNN model (ResNet-45) is pretrained on the ImageNet dataset (due to its 1000+ images each for an extensive list of classes) and the weights obtained are used in further training the ResNet model on the custom dataset.  -->

<div align="center">
  	<img src="Transfer Learning flowchart.drawio.png">
</div>

Using unsupervised learning, we can determine whether an item has expired or not by looking at images of the item. The full framework's architecture is depicted in Figure 1. Our framework is broken up into three sections. The feature extractor and the feature pyramid network (FPN) are used to extract the date region of the input image in the first section, which is referred to as date detection. The day, month, and year components are extracted from the date detection region, which is referred to as the Day-Month-Year Detection section. The third component is referred to as the Recognition Network, and this is where we apply the decoupling attention network (DAN) to recognize handwritten text. 

<div align="center">
  	<img src="ml_project.drawio (3).png">
</div>

# Results and Discussion

We used the following metrics for our classification:
1) Precision
2) Recall
3) F1 score
4) Accuracy

The confusion matrix for the classification problem in hand is as shown below:

*XXX confusion matrix image XXX*

The values of above reported metrics for our classification problem are as follows:
Precision = 
Recall = 
F1 score = 
Accuracy =

<!-- Results include a comparative analysis of all classification models (ResNet-45/50/101, VGG-16, Inceptionv3, EfficientNet) trained and tested for identifying expiry dates and classifying items.  

Performance metrics used to evaluate the models will be balanced accuracy, precision, recall, f1-score, confusion matrix, ROC AUC and Top-k classification accuracy. 

A mobile application that demonstrates the working of the models could also be designed. Further, this work can be extended towards the development of a scanner that can help the visually impaired in a potential lifesaving situation – as in the case of detecting expired medicines.  -->

# Reference
1. Ahmet Cagatay Seker, Sang Chul Ahn “A generalized framework for recognition of expiration dates on product packages using fully convolutional networks”, Expert Systems with Applications, Volume 203, 2022, 117310, ISSN 0957-4174, [https://doi.org/10.1016/j.eswa.2022.117310](https://doi.org/10.1016/j.eswa.2022.117310). 
2. Minyoung Huh, Pulkit Agrawal, Alexei A. Efros, “What makes ImageNet good for transfer learning?”, [https://doi.org/10.48550/arXiv.1608.08614](https://doi.org/10.48550/arXiv.1608.08614).
3. E. Peng, P. Peursum and L. Li, "Product Barcode and Expiry Date Detection for the Visually Impaired Using a Smartphone," 2012 International Conference on Digital Image Computing Techniques and Applications (DICTA), 2012, pp. 1-7, doi: 10.1109/DICTA.2012.6411673. 
4. Ashino, M., Takeuchi, Y. (2020). Expiry-Date Recognition System Using Combination of Deep Neural Networks for Visually Impaired. In: Miesenberger, K., Manduchi, R., Covarrubias Rodriguez, M., Peňáz, P. (eds) Computers Helping People with Special Needs. ICCHP 2020. Lecture Notes in Computer Science, vol 12376. Springer, Cham. [https://doi.org/10.1007/978-3-030-58796-3_58](https://doi.org/10.1007/978-3-030-58796-3_58).
5. Philipp Jund, Nichola Abdo, Andreas Eitel, Wolfram Burgard, "The Freiburg Groceries Dataset", https://doi.org/10.48550/arXiv.1611.05799
6. Mudrová, Martina and Aleš Procházka. “PRINCIPAL COMPONENT ANALYSIS IN IMAGE PROCESSING.” (2005).
7. Nsang, Augustine & Bello, A.M. & Shamsudeen, Hammed. (2015). Image reduction using assorted dimensionality reduction techniques. 1353. 139-146.
8. Patel, Chirag & Patel, Atul & Patel, Dharmendra. (2012). Optical Character Recognition by Open source OCR Tool Tesseract: A Case Study. International Journal of Computer Applications. 55. 50-56. 10.5120/8794-2784.
9. https://www.tensorflow.org/tutorials/images/classification 

# Gantt Chart and Proposed Timeline
[View File](https://gtvault-my.sharepoint.com/:x:/g/personal/rramachandra7_gatech_edu/Ecd-YPwCuFBDuvu44UX_7J0B0jfClvfIibe9kC5hi7yXXw?e=a2dXz0)

# Contribution Table
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-7btt">Name</th>
    <th class="tg-7btt">Project Task</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow" rowspan="3">Shivanee Jaiswal </td>
    <td class="tg-0pky">Project Midterm Report </td>
  </tr>
  <tr>
    <td class="tg-0pky">Unsupervised learning - Image compression </td>
  </tr>
  <tr>
    <td class="tg-0pky">Supervised Learning - Image date detection </td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="3">Reshma Ramachandra </td>
    <td class="tg-0pky">Project Midterm Report </td>
  </tr>
  <tr>
    <td class="tg-0pky">Unsupervised Learning - Product Classification </td>
  </tr>
  <tr>
    <td class="tg-0pky">Evaluation metrics </td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="3">Janavi Khochare </td>
    <td class="tg-0pky">Project Midterm Report </td>
  </tr>
  <tr>
    <td class="tg-0pky">Unsupervised learning - Image compression </td>
  </tr>
  <tr>
    <td class="tg-0pky">Supervised learning - Date detection </td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="3">Srikesh Reddy </td>
    <td class="tg-0pky">Project Midterm Report </td>
  </tr>
  <tr>
    <td class="tg-0pky">Data sourcing and cleaning for supervised learning </td>
  </tr>
  <tr>
    <td class="tg-0pky">Evaluation metrics </td>
  </tr>
  <tr>
    <td class="tg-c3ow" rowspan="3">Nikhil Viswanath </td>
    <td class="tg-0pky">Project Midterm Report </td>
  </tr>
  <tr>
    <td class="tg-0pky">Image augmentation </td>
  </tr>
  <tr>
    <td class="tg-0pky">Data sourcing and cleaning for supervised learning </td>
  </tr>
</tbody>
</table>
