<div align="center">
<h1>Automated Detection of Pre-training Text in Black-box LLMs</h1>
</div>

# 🔍VeilProbe

> This is the code for the paper.
>
> We propose VeilProbe, the framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. 
>
> <img src="/Users/mihuhu/Library/Application Support/typora-user-images/image-20250515040834089.png" alt="image-20250515040834089" style="zoom:80%;" />

It consists of three modules: 
 (a) The text-to-suffix sampling module generates text-to-suffix pairs; (b) The membership feature inference module infers membership features with a sequence-to-sequence model based on the above pairs; (c) The prototype-based classifier is trained based on the features from ground-truth samples, and then the pre-training and non-training prototypes are constructed. The first two modules prepare the membership feature inference stage for the texts to be detected, and the third one trains a membership classifier for detection.



## 💻 Reproduce our work

> We have uploaded the trained mapping model and the perturbed mapping model for direct reproduction on BookTection-128. 

```python
python run_protonet_classifier.py
```

> To reproduce results on other datasets, please first run the following command to generate the prefix-suffix pairs and perturbed prefix-suffix pairs.
>
> ```python
> python sampling_close_sourced_LLM.py #obtaining generated pairs
> python mappingmodel.py #obtaining mapping model
> ```

> To obtain the perturbed version, run 
>
> ```python
> keyperturb_agg.py
> ```
>
>  under the key_perturb directory.



## Citation
