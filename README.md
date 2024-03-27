# RNAdegformer

<p align="center">
  <img src="https://raw.githubusercontent.com/Shujun-He/RNAdegformer/main/graphics/RNAdegformer.png?token=GHSAT0AAAAAABRGHIRII6LFDFJ7KPWZWBEAYY6X33Q"/>
</p>




## Requirements
I included a file (environment.yml) to recreate the exact environment I used. Since I also use this environment for computer vision tasks, it includes some other packages as well. This should take around 10 minutes. After installing anaconda:


```
conda env create -f environment.yml
```

Then to activate the environment

```
conda activate torch
```

Additionally, you will need Nvidai Apex: https://github.com/NVIDIA/apex

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install .
```

Also you need to install the Ranger optimizer

```bash
git clone https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
cd Ranger-Deep-Learning-Optimizer
pip install -e .
```


