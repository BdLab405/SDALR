# Official implementation for **SDALR**

## [**Source-free domain adaptation based on label reliability for cross-domain bearing fault diagnosis**](http://www.baidu.com)



### Framework:  

<img src="/fig/SDALR.jpg" width="800"/>

### Prerequisites:
- python == 3.11.10
- pytorch ==2.4.1
- numpy, scipy, sklearn, PIL, argparse, tqdm, openpyxl, collections

### Dataset:

-  Please manually download the datasets [PU](https://pan.baidu.com/s/1d505GjqsmHWlwFG5hb5c3Q?pwd=5m1l), [JNU](https://pan.baidu.com/s/1d505GjqsmHWlwFG5hb5c3Q?pwd=5m1l),
  
-  Concerning the dsatasets, put it into './DATA/'.


### Experimental Results

#### Diagnosis Accuracy
We evaluate SDALR on the PU and JNU datasets under source-free settings.  
The results show that SDALR consistently outperforms UDA and SFDA baselines.  

👉 Detailed per-task results can be found in the [paper](link).

#### Reliability Threshold ($\partial$):
  With $\beta=0.6$, we varied $\partial$ from 0.45 to 0.95.  
  - Too high ($\partial > 0.9$): pseudo-label accuracy fluctuates, unreliable samples increase.  
  - Moderate ($0.4 \leq \partial \leq 0.7$): stable pseudo-label accuracy and diagnostic performance.  
  - Conclusion: $\partial$ governs the trade-off, while voting ensures robustness.  

<table>
  <tr>
    <td><img src="/fig/fig1.png" width="400"></td>
    <td><img src="/fig/fig2.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="/fig/fig3.png" width="400"></td>
    <td><img src="/fig/fig4.png" width="400"></td>
  </tr>
</table>

#### Sample Size Analysis:
  With $\beta=0.6$, $\partial=0.6$, we varied the number of target samples.  
  - (a) Accuracy of all pseudo-labels (%)  
  - (b) Fault diagnosis accuracy (%)  
  - Results confirm the scalability and robustness of the method.

<table>
  <tr>
    <td><img src="/fig/fig5.png" width="400"></td>
    <td><img src="/fig/fig6.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="/fig/fig7.png" width="400"></td>
    <td><img src="/fig/fig8.png" width="400"></td>
  </tr>
</table>

### Training

1. ##### Source-free Domain Adaptation (SFDA) on PU dataset
   - Pre-train the source model:
     ```bash
     cd object/
     python src_pretrain_RES_PU.py
     ```
   - Adaptation to the target domain:  
     We provide a shell script for automatic adaptation.  
     The script supports **custom arguments** (e.g., number of epochs, batch size, learning rate) that can be modified directly in the `.sh` file for flexible control.  
     ```bash
     ./choose_s_t_PU.sh
     ```

2. ##### Source-free Domain Adaptation (SFDA) on JNU dataset
   - Pre-train the source model:
     ```bash
     cd object/
     python src_pretrain_RES_JNU.py
     ```
   - Adaptation to the target domain:  
     Similar to PU, the script allows users to **customize various parameters** inside the `.sh` file for automated adaptation.  
     ```bash
     ./choose_s_t_JNU.sh
     ```

3. ##### Ablation Studies
   To reproduce the ablation experiments reported in the paper, simply run:
   ```bash
   ./ablation_PU.sh
   ./ablation_JNU.sh

   	 ```

### Contact

- [bd_lab@163.com](mailto:bd_lab@163.com)
