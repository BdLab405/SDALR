# Official implementation for **SDALR**

## [**Source-free domain adaptation based on label reliability for cross-domain bearing fault diagnosis**](http://www.baidu.com)



### Framework:  

<img src="SDALR.jpg" width="800"/>

### Prerequisites:
- python == 3.11.10
- pytorch ==2.4.1
- numpy, scipy, sklearn, PIL, argparse, tqdm, openpyxl, collections

### Dataset:

-  Please manually download the datasets [PU](https://pan.baidu.com/s/1d505GjqsmHWlwFG5hb5c3Q?pwd=5m1l), [JNU](https://pan.baidu.com/s/1d505GjqsmHWlwFG5hb5c3Q?pwd=5m1l),
  
-  Concerning the dsatasets, put it into './DATA/'.


### Experimental Results
- **Reliability Threshold ($\partial$):**  
  With $\beta=0.6$, we varied $\partial$ from 0.45 to 0.95.  
  - Too high ($\partial > 0.9$): pseudo-label accuracy fluctuates, unreliable samples increase.  
  - Moderate ($0.4 \leq \partial \leq 0.7$): stable pseudo-label accuracy and diagnostic performance.  
  - Conclusion: $\partial$ governs the trade-off, while voting ensures robustness.  

<table>
  <tr>
    <td><img src="fig1.png" width="300"></td>
    <td><img src="fig2.png" width="300"></td>
  </tr>
  <tr>
    <td><img src="fig3.png" width="300"></td>
    <td><img src="fig4.png" width="300"></td>
  </tr>
</table>

- **Sample Size Analysis:**  
  With $\beta=0.6$, $\partial=0.6$, we varied the number of target samples.  
  - (a) Accuracy of all pseudo-labels (%)  
  - (b) Fault diagnosis accuracy (%)  
  - Results confirm the scalability and robustness of the method.

<table>
  <tr>
    <td><img src="fig5.png" width="300"></td>
    <td><img src="fig6.png" width="300"></td>
  </tr>
  <tr>
    <td><img src="fig7.png" width="300"></td>
    <td><img src="fig8.png" width="300"></td>
  </tr>
</table>

### Training:
1. ##### Source-free Domain Adaptation (SFDA) on the dataset PU
	- Train model on the source domains, respectively
	```python
	 cd object/
	 python src_pretrain_RES_PU.py
	```
	
	- Adaptation to the target domain
	```python
	./choose_s_t_PU.sh
	```
	
2. ##### Source-free Domain Adaptation (SFDA) on the dataset JNU
	- Train model on the source domains, respectively
	```python
	 cd object/
	 python src_pretrain_RES_JNU.py
	```
	
	- Adaptation to the target domain
	```python
	 ./choose_s_t_JNU.sh
   	 ```

### Contact

- [bd_lab@163.com](mailto:bd_lab@163.com)
