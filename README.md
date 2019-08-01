# LSTM-GAN-MNIST

## Link to the paper, visit: [Link](https://link.springer.com/chapter/10.1007/978-981-13-1610-4_11)

## To easily cite our Paper, visit : [Link](https://link.springer.com/chapter/10.1007/978-981-13-1610-4_11#citeas)

## Brief description: 

Using the MNIST set to experiment with GANs using LSTM's 

![alt text](https://github.com/amitadate/S-LSTM-GAN-MNIST/blob/master/images/faster_gif.gif)

Model
=====
Following a generic generative adversarial network, the model consists two networks trained in parallel, and sharing weights.
The pink portion of the model is the generator and the orange-brown portion is the discriminator. For purposes of clarity the image is
split into quadrants here, but in other experiments the attempt was to split the image into pixels in an attempt to create a 
generator that could create digits pixel by pixel using long range memory. Up to now the best results have occurred with splitting
the image into 16 sections, beyond that the model fails.

![alt text](https://github.com/amitadate/S-LSTM-GAN-MNIST/blob/master/images/model_diagram.jpg)

Generator
---------
![alt text](https://github.com/amitadate/S-LSTM-GAN-MNIST/blob/master/images/model_diagram_gen.jpg)

Discriminator
---------
![alt text](https://github.com/amitadate/S-LSTM-GAN-MNIST/blob/master/images/model_diagram_disc.jpg)

Experiments
=====

### TIMESTEP MODEL

| Variable          | Value     |
| :---------------- | :---------|
| timesteps         | 4         |
| lstm_layers_RNN_g | 6        |
| lstm_layers_RNN_d | 2         |
| hidden_size_RNN_g | 600       |
| hidden_size_RNN_d | 400       |
| lr                | 1e-4    |
| iterations        | > 2.5e6       |

#### SAMPLES

|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|![alt tag](atsteps4/im0_1.png)|![alt tag](atsteps4/im1_1.png)|![alt tag](atsteps4/im2_1.png)|![alt tag](atsteps4/im3_1.png)|![alt tag](atsteps4/im4_1.png)|![alt tag](atsteps4/im5_1.png)|![alt tag](atsteps4/im6_1.png)|![alt tag](atsteps4/im7_1.png)|![alt tag](atsteps4/im8_1.png)|![alt tag](atsteps4/im9_1.png)|
|![alt tag](atsteps4/im0_2.png)|![alt tag](atsteps4/im1_2.png)|![alt tag](atsteps4/im2_2.png)|![alt tag](atsteps4/im3_2.png)|![alt tag](atsteps4/im4_2.png)|![alt tag](atsteps4/im5_2.png)|![alt tag](atsteps4/im6_2.png)|![alt tag](atsteps4/im7_2.png)|![alt tag](atsteps4/im8_2.png)|![alt tag](atsteps4/im9_2.png)|
|![alt tag](atsteps4/im0_3.png)|![alt tag](atsteps4/im1_3.png)|![alt tag](atsteps4/im2_3.png)|![alt tag](atsteps4/im3_3.png)|![alt tag](atsteps4/im4_3.png)|![alt tag](atsteps4/im5_3.png)|![alt tag](atsteps4/im6_3.png)|![alt tag](atsteps4/im7_3.png)|![alt tag](atsteps4/im8_3.png)|![alt tag](atsteps4/im9_3.png)|
|![alt tag](atsteps4/im0_4.png)|![alt tag](atsteps4/im1_4.png)|![alt tag](atsteps4/im2_4.png)|![alt tag](atsteps4/im3_4.png)|![alt tag](atsteps4/im4_4.png)|![alt tag](atsteps4/im5_4.png)|![alt tag](atsteps4/im6_4.png)|![alt tag](atsteps4/im7_4.png)|![alt tag](atsteps4/im8_4.png)|![alt tag](atsteps4/im9_4.png)|
|![alt tag](atsteps4/im0_5.png)|![alt tag](atsteps4/im1_5.png)|![alt tag](atsteps4/im2_5.png)|![alt tag](atsteps4/im3_5.png)|![alt tag](atsteps4/im4_5.png)|![alt tag](atsteps4/im5_5.png)|![alt tag](atsteps4/im6_5.png)|![alt tag](atsteps4/im7_5.png)|![alt tag](atsteps4/im8_5.png)|![alt tag](atsteps4/im9_5.png)|
|![alt tag](atsteps4/im0_6.png)|![alt tag](atsteps4/im1_6.png)|![alt tag](atsteps4/im2_6.png)|![alt tag](atsteps4/im3_6.png)|![alt tag](atsteps4/im4_6.png)|![alt tag](atsteps4/im5_6.png)|![alt tag](atsteps4/im6_6.png)|![alt tag](atsteps4/im7_6.png)|![alt tag](atsteps4/im8_6.png)|![alt tag](atsteps4/im9_6.png)|
|![alt tag](atsteps4/im0_7.png)|![alt tag](atsteps4/im1_7.png)|![alt tag](atsteps4/im2_7.png)|![alt tag](atsteps4/im3_7.png)|![alt tag](atsteps4/im4_7.png)|![alt tag](atsteps4/im5_7.png)|![alt tag](atsteps4/im6_7.png)|![alt tag](atsteps4/im7_7.png)|![alt tag](atsteps4/im8_7.png)|![alt tag](atsteps4/im9_7.png)|
|![alt tag](atsteps4/im0_8.png)|![alt tag](atsteps4/im1_8.png)|![alt tag](atsteps4/im2_8.png)|![alt tag](atsteps4/im3_8.png)|![alt tag](atsteps4/im4_8.png)|![alt tag](atsteps4/im5_8.png)|![alt tag](atsteps4/im6_8.png)|![alt tag](atsteps4/im7_8.png)|![alt tag](atsteps4/im8_8.png)|![alt tag](atsteps4/im9_8.png)|
|![alt tag](atsteps4/im0_9.png)|![alt tag](atsteps4/im1_9.png)|![alt tag](atsteps4/im2_9.png)|![alt tag](atsteps4/im3_9.png)|![alt tag](atsteps4/im4_9.png)|![alt tag](atsteps4/im5_9.png)|![alt tag](atsteps4/im6_9.png)|![alt tag](atsteps4/im7_9.png)|![alt tag](atsteps4/im8_9.png)|![alt tag](atsteps4/im9_9.png)|
|![alt tag](atsteps4/im0_10.png)|![alt tag](atsteps4/im1_10.png)|![alt tag](atsteps4/im2_10.png)|![alt tag](atsteps4/im3_10.png)|![alt tag](atsteps4/im4_10.png)|![alt tag](atsteps4/im5_10.png)|![alt tag](atsteps4/im6_10.png)|![alt tag](atsteps4/im7_10.png)|![alt tag](atsteps4/im8_10.png)|![alt tag](atsteps4/im9_10.png)|

![alt tag](atsteps4/loss_full_aug_8.png)

![alt tag](atsteps4/classification_full_aug_8.png)

### TIMESTEP MODEL

| Variable          | Value     |
| :---------------- | :---------|
| timesteps         | 16         |
| lstm_layers_RNN_g | 6        |
| lstm_layers_RNN_d | 2         |
| hidden_size_RNN_g | 600       |
| hidden_size_RNN_d | 400       |
| lr                | 2e-4:GEN/1e-4:DISC    |
| iterations        | > 5e5       |

#### SAMPLES

|0|1|2|3|4|5|6|7|8|9|
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
|![alt tag](atsteps16/im0_0.png)|![alt tag](atsteps16/im1_0.png)|![alt tag](atsteps16/im2_0.png)|![alt tag](atsteps16/im3_0.png)|![alt tag](atsteps16/im4_0.png)|![alt tag](atsteps16/im5_0.png)|![alt tag](atsteps16/im6_0.png)|![alt tag](atsteps16/im7_0.png)|![alt tag](atsteps16/im8_0.png)|![alt tag](atsteps16/im9_0.png)|
|![alt tag](atsteps16/im0_1.png)|![alt tag](atsteps16/im1_1.png)|![alt tag](atsteps16/im2_1.png)|![alt tag](atsteps16/im3_1.png)|![alt tag](atsteps16/im4_1.png)|![alt tag](atsteps16/im5_1.png)|![alt tag](atsteps16/im6_1.png)|![alt tag](atsteps16/im7_1.png)|![alt tag](atsteps16/im8_1.png)|![alt tag](atsteps16/im9_1.png)|
|![alt tag](atsteps16/im0_2.png)|![alt tag](atsteps16/im1_2.png)|![alt tag](atsteps16/im2_2.png)|![alt tag](atsteps16/im3_2.png)|![alt tag](atsteps16/im4_2.png)|![alt tag](atsteps16/im5_2.png)|![alt tag](atsteps16/im6_2.png)|![alt tag](atsteps16/im7_2.png)|![alt tag](atsteps16/im8_2.png)|![alt tag](atsteps16/im9_2.png)|
|![alt tag](atsteps16/im0_3.png)|![alt tag](atsteps16/im1_3.png)|![alt tag](atsteps16/im2_3.png)|![alt tag](atsteps16/im3_3.png)|![alt tag](atsteps16/im4_3.png)|![alt tag](atsteps16/im5_3.png)|![alt tag](atsteps16/im6_3.png)|![alt tag](atsteps16/im7_3.png)|![alt tag](atsteps16/im8_3.png)|![alt tag](atsteps16/im9_3.png)|
|![alt tag](atsteps16/im0_4.png)|![alt tag](atsteps16/im1_4.png)|![alt tag](atsteps16/im2_4.png)|![alt tag](atsteps16/im3_4.png)|![alt tag](atsteps16/im4_4.png)|![alt tag](atsteps16/im5_4.png)|![alt tag](atsteps16/im6_4.png)|![alt tag](atsteps16/im7_4.png)|![alt tag](atsteps16/im8_4.png)|![alt tag](atsteps16/im9_4.png)|
|![alt tag](atsteps16/im0_5.png)|![alt tag](atsteps16/im1_5.png)|![alt tag](atsteps16/im2_5.png)|![alt tag](atsteps16/im3_5.png)|![alt tag](atsteps16/im4_5.png)|![alt tag](atsteps16/im5_5.png)|![alt tag](atsteps16/im6_5.png)|![alt tag](atsteps16/im7_5.png)|![alt tag](atsteps16/im8_5.png)|![alt tag](atsteps16/im9_5.png)|
|![alt tag](atsteps16/im0_6.png)|![alt tag](atsteps16/im1_6.png)|![alt tag](atsteps16/im2_6.png)|![alt tag](atsteps16/im3_6.png)|![alt tag](atsteps16/im4_6.png)|![alt tag](atsteps16/im5_6.png)|![alt tag](atsteps16/im6_6.png)|![alt tag](atsteps16/im7_6.png)|![alt tag](atsteps16/im8_6.png)|![alt tag](atsteps16/im9_6.png)|
|![alt tag](atsteps16/im0_7.png)|![alt tag](atsteps16/im1_7.png)|![alt tag](atsteps16/im2_7.png)|![alt tag](atsteps16/im3_7.png)|![alt tag](atsteps16/im4_7.png)|![alt tag](atsteps16/im5_7.png)|![alt tag](atsteps16/im6_7.png)|![alt tag](atsteps16/im7_7.png)|![alt tag](atsteps16/im8_7.png)|![alt tag](atsteps16/im9_7.png)|
|![alt tag](atsteps16/im0_8.png)|![alt tag](atsteps16/im1_8.png)|![alt tag](atsteps16/im2_8.png)|![alt tag](atsteps16/im3_8.png)|![alt tag](atsteps16/im4_8.png)|![alt tag](atsteps16/im5_8.png)|![alt tag](atsteps16/im6_8.png)|![alt tag](atsteps16/im7_8.png)|![alt tag](atsteps16/im8_8.png)|![alt tag](atsteps16/im9_8.png)|
|![alt tag](atsteps16/im0_9.png)|![alt tag](atsteps16/im1_9.png)|![alt tag](atsteps16/im2_9.png)|![alt tag](atsteps16/im3_9.png)|![alt tag](atsteps16/im4_9.png)|![alt tag](atsteps16/im5_9.png)|![alt tag](atsteps16/im6_9.png)|![alt tag](atsteps16/im7_9.png)|![alt tag](atsteps16/im8_9.png)|![alt tag](atsteps16/im9_9.png)|

![alt tag](atsteps16/loss_sep_4_18.png)

![alt tag](atsteps16/classification_sep_4_18.png)
