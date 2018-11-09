

# A Survey of Recent Advances in CNN-based Single Image Crowd Counting and Density Estimation

### Outline: 

1. Methods rely largely on hand-crafted representations.
2. Deep learning-based approaches.
3. Recently published datasets.
4. Merits of drawbacks of existing CNN-based approaches.
5. Promising avenues.



**Dilemma**: High clutter, overlapping, variation in appearance, scale , perspective and distribution of people and other properties such as illumination.



### Traditional Methods:

1. #### Detection-based:

   1. monolithic style detection.

      + Typically traditional methods which trains a classifier like SVM, Boosting, RF, using hand-crafted features, such as Haar, HOG, etc.

      + Pros & Cons: successful in low density crowd scenes, while adversely affected by the presence of high density crowds.

   2. parts-based detection.

      + Adopting part-based detection methods -- boosted classifier for specific body parts, shape learning.

2. #### Regression-based:

   1. Learn a mapping between features extracted from local image patches to their counts -- low-level feature(edge, texture, gradient, etc) extraction and regression modelling.

   2. Pros & Cons: avoid dependency on detectors, inherently capable of handling imbalanced data.

   3. An example method by Idrees _et al._:

      Fourier analysis + SIFT interest points + head detection -- into --> Markov Random Field -- output --> a cumulative attribute space.

3. #### Density estimation-based:

   1. Improvement: While earlier methods are good at dealing with occlusion and clutter, they were regressing on the global count, ignoring import spatial information. 
   2. Progress 1(linear) by Lempitsky _et al._:
      + Method: **object_density_maps = a_linear_mapping(local_path_features)**, where the integral value of a certain area is the count of objects in it. After that, the problem of learning density maps is formulated as a new regularized risk quadratic cost function, which they solved using cutting-lane optimization.
   3. Progress 2(non-linear) by Pham _et al._:
      + Method: Use **random forest regression** from multiple image patches to vote for density map of objects. To tackle the problem of large variation between crowded patched and non-crowded ones, a crowdedness prior is proposed and two different forests corresponding to this prior are trained.
   4. Progress 3(better performance in computational complexity) by Wang and Zou.
      + Method: Propose a subspace learning-based density estimation to compute the embedding of each  subspace formed by image patches, then learn the mapping between the density map of input and corresponding embedding matrix. In the computation, they speed it up by dividing the feature spaces of image patches and their counterpart density maps into subspaces.
      + Assumption: local image patches and their corresponding density maps share similar local geometry.
   5. Progress 4 by Xu and Qiu:
      + Method: Inspired by the high-dimensional features in other domains, they use a much extensive and richer set of features, and under the limit of that era when regression is under curse of dimensionality, they embedded random projection in the tree nodes to construct random forest, which also introduce randomness in the tree construction.



### CNN-based Methods:

1. #### Category:

   + ##### In terms of Network Property:

     1. ###### Basic CNNs:

        ​	Initial deep learning approaches for crowd counting and density estimation.

     2. ###### Scale-aware models:

        ​	Achieve better robustness to variations in scale by techniques such as multi-column or multi-resolution architectures.

     3. ###### Context-aware models:

        ​	Incorporate local and global contextual information.

     4. ###### Multi-task frameworks:

        ​	Combine crowd counting and estimation along with other tasks such as foreground-background subtraction and crowd velocity estimation.

   + ##### In terms of Training process:

     1. ###### Patch-based inference:

        ​	Use patches cropped from the input images whose size differ among methods. During the prediction phase, a sliding window is used here and predictions from all windows finally aggregated to sum up.

     2. ###### Whole image-based inference:

        ​	As its name, these method avoid computationally expensive sliding windows.

2. #### Methods:

   1. ##### An basic end-to-end deep CNN regression model, by Wang _et al._ and Fu _et al._:

      + Network:
        1. Adopt AlexNet, replace the final 4096 FC later with a single neuron layer for predicting the count.
        2. Fu _et al._ proposed to classify the image into one of the five classes: very high, high ... instead of density maps.
      + Data augment:
        1. Training data is augmented with additional negative samples whose ground truth count is set as zero to reduce false responses background.

   2. ##### Multi-stage ConvNet, by Sermanet _et al_:

      + Improvements:

        ​	Better shift, scale and distortion invariance.

      + Network:

        ​	Use a cascade of two classifiers to achieve boosting where the former one specially samples misclassified image and the latter one reclassifies rejected samples.

   3. ##### Cross-scene counting(apply a mapping learned into new target scenes), by Zhang _et al._:

      + Network:

        1. Training: Learn the network by alternatively training on two objective functions: crowd count and density estimation -- X: Crowd patches, Y: Crowd counts / Density patches.
        2. Fine-tuning to a new scene: Use training samples that are similar to the target scene.

      + Improvements:

        ​	Substitute Gaussian kernels which are used for generating ground truth density map with a new method that can incorporate perspective information in the generated map.

   4. ##### Layered boosting and selective sampling, by Walach and Wolf:

      + Inspired by Cross-scene counting and Gradient Boosting Machines(GBM).

      + Layered boosting: iteratively adding CNN layers to model -- (n+1)-th CNN layer is trained on the difference between the estimation of n-th CNN layer and ground truth.
      + Selective sampling: samples that are correctly classified early on are trivial samples, which tend to introduce bias in the network for such samples and therefore affecting its generalization performance.

   5. ##### End-to-end count estimation taking the entire image, by Shang _et al._:

      + Network:

        ```flow
        input=>inputoutput: Images
        feature_extractor=>operation: GoogLeNet(feature_extractor)
        decoder=>operation: LSTM(decoder, output=local counts)
        FC_layers=>operation: FC_layers(output=global counts)
        output=>inputoutput: Final output
        
        input(right)->feature_extractor()->decoder(right)->FC_layers(right)->output
        ```

   6. ##### Combine deep and shallow fully convolutional networks, by Boominathan _et al._:

      + Network:

        ​	The combination of both deep and shallow FCN makes the network more robust to  non-uniform scaling of crowd and variations in perspective.

        ```flow
        input=>inputoutput: Crowd Image
        cond=>condition: Shallow?
        conv_1=>operation: conv(3x3)*2
        conv_2=>operation: conv(3x3)*2
        conv3_1=>operation: conv(3x3)*3
        conv3_2=>operation: conv(3x3)*3
        conv3_3=>operation: conv(3x3)*3
        max_pooling2_1=>operation: Max Pool 2x2
        max_pooling2_2=>operation: Max Pool 2x2
        max_pooling2_3=>operation: Max Pool 2x2
        max_pooling3_1=>operation: Max Pool 3x3
        conv5_1=>operation: conv(5x5)*1
        max_pooling5_1=>operation: Max Pool 5x5
        conv5_2=>operation: conv(5x5)*1
        max_pooling5_2=>operation: Max Pool 5x5
        conv5_3=>operation: conv(5x5)*1
        max_pooling5_3=>operation: Max Pool 5x5
        concat=>operation: Concat
        conv1=>operation: conv(1x1)*1
        interp=>operation: Interpolation
        EDP=>inputoutput: Estimated Density Map
        sum=>operation: Sum up
        output=>inputoutput: Estimated Count
        
        input(right)->cond
        cond(yes)->conv_1(right)->max_pooling2_1(right)->conv_2(right)->max_pooling2_2(right)->conv3_1(right)->max_pooling2_3(right)->conv3_2(right)->max_pooling3_1(right)->conv3_3(right)->concat
        cond(no)->conv5_1(right)->max_pooling5_1(right)->conv5_2(right)->max_pooling5_2(right)->conv5_3(right)->max_pooling5_3(right)->concat
        concat(right)->conv1(right)->interp(right)->EDP->sum->output
        ```

      + Data augmentation:

        ​	Sample patches from the multi-scale image representation to make model more robust to scale variations.

   7. ##### Multi-column CNN for images with arbitrary crowd density and arbitrary perspective, by Zhang _et al._:

      + Network:

        ​	Comprises of three columns corresponding to filters with receptive fields of different sizes.

      + Improvement:

        ​	Take into account perspective distortion by estimating spread parameter of the Gaussian kernel based on the size of the head of each person within the image, and the spread parameter for each person is data-adaptively determined based on its average distance to its neighbors.

      + Dataset:

        ​	Contribute the ShanghaiTech crowd datasets.

   8. ##### Hydra CNN, by Onoro and Sastre:

      + 















# Multi-column CNN

\* 大多数情况都是预测密度分布图的.

### 针对头像大小不一样:

+ 传统方法:
  1. 不同长宽大小的window进行slide;
  2. 同样规格的window, 然先通过图像金字塔得到不同尺寸的图像, 再进行slide;
+ DL:
  1. 对应于传统方法1, 让网络分流, 不同的流利用不同大小的滤波器进行卷积, 如: Multi-column CNN.