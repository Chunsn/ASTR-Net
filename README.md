```markdown
# This is the code repository for the paper "ASTR-Net: Attentive Spatial-Temporal Refinement Network for Personalized EEG Source Imaging".

![ASTR-Net Network Architecture](images/fig2.png)

The repository provides the required network architecture and main function code.

1. The lead field array and simulation data used for model pre-training are based on the dataset provided by DeepSIF. [Please visit](https://github.com/bfinl/DeepSIF). We cite this work in this paper and express our gratitude herein.

2. The main function for conducting pre-training is `main_Pre_training.py`, and the dataset used is `leadfield_75_20k.mat`. The pre-training employs an early stopping mechanism.

3. **Personalized Lead Field Generation**

   To generate patient-specific head models, first download the synchronized EEG–iEEG dataset from [OSF](https://doi.org/10.17605/OSF.IO/WSGZP).

   Then, use **BrainSuite** to perform MRI segmentation and construct the personalized head model for each patient.

   After segmentation, run the following MATLAB script located in the `personalized_leadfield` folder:

   ```matlab
   run_pilot_personalized_leadfields.m
