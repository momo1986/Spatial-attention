# Spatial attention 
vehicle trajectory prediction

## Enviorment requirement

python 3.7.3

pytorch 1.1.0 --gpu

sklearn 0.20.3 

pynvml 8.0.1

---------------------

## code in Enocder-Decoder Net folder

main.py

after run main.py, you will get a output document like slurm xxxx.out

please put the data in the same fold as main.py

lane attention.py: code for lane-attention 

context attention.py: code for context-attention

soft attention.py: code for soft-attention

baseline.py: code for the base structure without attention

Supplementary beam search program: main_supplement_beam_search.py 
Supplementary social polling program main_scocial_pooling.py

-------------------- 
## datasets
1. driving sequence data: 4/6track_input_c.npy(the first 3 seconds of all surrounding 9 vehicles), 
                                         4/6position_output.npy(the last 5 seconds of target car position)  

2. data_split.py is for spliting origin data into train,validation,test datasets


-------------------
## result show

rmse_compare.py
