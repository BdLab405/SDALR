# Official implementation for **SDALR**

## [**Source-free domain adaptation based on label reliability for cross-domain bearing fault diagnosis**](http://www.baidu.com)



### Framework:  

<img src="SDALR.jpg" width="600"/>

### Prerequisites:
- python == 3.6.8
- pytorch ==1.1.0
- torchvision == 0.3.0
- numpy, scipy, sklearn, PIL, argparse, tqdm

### Dataset:

-  Please manually download the datasets [PU](https://pan.baidu.com/s/1d505GjqsmHWlwFG5hb5c3Q?pwd=5m1l), [JNU](https://pan.baidu.com/s/1d505GjqsmHWlwFG5hb5c3Q?pwd=5m1l),
  
-  Concerning the dsatasets, put it into './DATA/'.


### Training:
1. ##### Source-free Domain Adaptation (SFDA) on the dataset PU
	- Train model on the source domains, respectively
	```python
	 cd object/
	 python src_pretrain_RES.py --gpu_id 0 --max_epoch 10 --interval 2 --username PU --data_name PU_1d_8c_2048 --domain_names ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04'] --class_num 8 --s 0
	 python src_pretrain_RES.py --gpu_id 0 --max_epoch 10 --interval 2 --username PU --data_name PU_1d_8c_2048 --domain_names ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04'] --class_num 8 --s 1
	 python src_pretrain_RES.py --gpu_id 0 --max_epoch 10 --interval 2 --username PU --data_name PU_1d_8c_2048 --domain_names ['N15_M01_F10', 'N15_M07_F10', 'N15_M07_F04'] --class_num 8 --s 2
	```
	
	- Adaptation to the target domain
	```python
	 python tar_adaptation_RES_PU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 0 --t 1
	 python tar_adaptation_RES_PU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 0 --t 2
 	 python tar_adaptation_RES_PU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 1 --t 0
 	 python tar_adaptation_RES_PU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 1 --t 2
 	 python tar_adaptation_RES_PU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 2 --t 0
 	 python tar_adaptation_RES_PU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 2 --t 1
	```
	
2. ##### Source-free Domain Adaptation (SFDA) on the dataset JNU
	- Train model on the source domains, respectively
	```python
	 cd object/
	 python src_pretrain_RES.py --gpu_id 0 --max_epoch 10 --interval 2 --username JUN --data_name JNU_1d_2048_2000 --domain_names ['600', '800', '1000'] --class_num 8 --s 0
	 python src_pretrain_RES.py --gpu_id 0 --max_epoch 10 --interval 2 --username JUN --data_name JNU_1d_2048_2000 --domain_names ['600', '800', '1000'] --class_num 8 --s 1
	 python src_pretrain_RES.py --gpu_id 0 --max_epoch 10 --interval 2 --username JUN --data_name JNU_1d_2048_2000 --domain_names ['600', '800', '1000'] --class_num 8 --s 2
	```
	
	- Adaptation to the target domain
	```python
	 python tar_adaptation_RES_JNU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 0 --t 1
	 python tar_adaptation_RES_JNU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 0 --t 2
 	 python tar_adaptation_RES_JNU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 1 --t 0
 	 python tar_adaptation_RES_JNU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 1 --t 2
 	 python tar_adaptation_RES_JNU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 2 --t 0
 	 python tar_adaptation_RES_JNU.py  --gpu_id 0 --max_epoch 20 --interval 4 --s 2 --t 1
   	 ```

**Please refer *./object/run.sh*** for all the settings for different methods and scenarios.

### Contact

- [bd_lab@163.com](mailto:bd_lab@163.com)
