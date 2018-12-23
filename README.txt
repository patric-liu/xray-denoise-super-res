Project group:  Patric Liu (pfliu2) and David Yang (haozey2)

Preprocessing:

Our approach used very straightforward supervised learning, so the only preprocessing used in our final approach was loading and greyscaling the images.


Approach:

Vanilla supervised learning: (64,64,1) shape input and (128,128,1) shape  targets. Final model was trained on all the data with ~30 epochs. 


Architecture:

We used a deep residual neural network. 3x3 convolutions, 32 filters for every layer. 5 64x64 residual layers, a transpose convolution layer, then 5 more 128x128 residual layers. We chose to use residual blocks because it seems to naturally lend itself to these types of problems, since the input is extremely similar to the output. We're not sure if/how much the transpose convolution layer reduced the effectiveness of residual layers, since there was no simple way to have a residual transpose convolution layer. All activations were relu, and the output layer had a sigmoid activation to constrain values to between 0 and 1 (which are then scaled up to [0,255]).

Experiments with leaky relu as well as hard sigmoid activations didn't show much improvement. Never got the time to try regularization or other optimizers. I also tried a variation of the residual block where the skip connection was added back after the second activation, but that slowed learning. 


Hyper-Parameters:

We used the adadelta optimizer which required no learning rate tuning or scheduling. No regularization or dropout was used. VERY interestingly, using mean squared error lead to both lower RMSE training error and better generalization than using RMSE error. Training alternated between epochs of RMSE and MSE losses before finishing off with MSE. A paper by nvidia a while found (by alternativing l1 and l2 error) that this can help get the network out of local minima and it seemed to work. 


Inspiration: 

It was the most simple yet feasible architecture and training method that we could think of, since we did not have time to deal with more complex approaches.


###### NOTE ########

Our architecture achieved a 'high' score of ~8700 on the validation set. We found that most of the 64x64 test images could be found in the training set (and messaged the staff about it), so we used the corresponding 128x128 images from the training set in our final submission, hence the sub-4000 score. If you change your mind about this being allowed, we'd be happy to give you our original submission which achieved 8700.



