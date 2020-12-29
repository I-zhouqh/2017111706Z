## Deep clustering


Self-supervised learning using deep clustering as algorithm and vgg as convnet can be launched by running

```
deepcluster.py
```
and then several models will be saved in /model_save_time directory. You can run
```
eval_linear.py
```
to check the ability of feature extraction of them. What you need to change is to specify the model name in "model_list" variable.
the file of "parameter_epoch_40.pkl" is an example of the parameter.

## Semi-supervised learning

Using features extracted by deep clusterings to do semi-supervised learning (the pseudo labels method), you can run
```
SSL.py
```
what you need do change is 'frozen_model_path' and 'frozen_model_conv' keywords in variable 'args', indicating the path of the 
deep clustering model and the layer to use seperately. Usually the chosen model and chosen layer is determined by the result of eval_linear.py.
The parameter for this model is stored in model/deepclustering_with_SSL.pkl


## Methods for comparision

1. Null model can trained by run 
```
Null_model/main.py
```
and the parameter is stored in this model/null_model.pkl.

2. Deep clustering model only without semi-supervised learning can be trained by run
```
SSL.py
```
and set args['max_alpha'] to zero. The parameter is stored in model/deep_clustering_only.pkl.

3. Semi-supervised learning without using deep clustering can be launched by run
```
Semi-supervised model/SSL_nopretrain.py
```
The parameter is stored in model/SSL_only.pkl.

## Predict on 49500 samples using trained model

This is our final step. In order to evaluate the prediction accuracy on 49500 samples, run
```
test_on_all_49500.py
```
Note that you should set args['model_path'] as the path of the model you want to use,
when you want to evaluate "SSL only model" or "Null model", set args['net_type']='vgg',
when you want to evaluate "Deep clustering model" or "SSL with Deep clusering model", set args['net_type']='deep_clustered_vgg'.


