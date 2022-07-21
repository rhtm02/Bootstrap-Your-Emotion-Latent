# The 4th Workshop and Competition on Affective Behavior Analysis in-the-wild (ABAW), will be held in conjunction with the European Conference on Computer Vision (ECCV), 2022. 


## Team : IMLAB

BYEL(Bootstrap on Your Emotion Latent) - https://arxiv.org/abs/2207.10003


## Requirements
- Anaconda must be installed and GPU-enabled.
- conda_requirements.yml should be fixed to fit your anaconda path.

 - Then, you can create new conda environment.
    
    ```python
    $ conda env create -f imlab.yml
    $ conda activate abaw
    $ pip install -r dependency.txt
    ```
## Dataset

1. Download data in ./dataset directory(dataset/)
    - You should download idx pickle including file_name, labels etc.   
    ```markdown
    dataset
    	-train
    	-evaluation
    ```
    
## Acknowledgement

This code was modified by taking a lot of the code from https://github.com/lucidrains/byol-pytorch.



## Citation
@misc{lee2022byel,
      title={BYEL : Bootstrap on Your Emotion Latent}, 
      author={Hyungjun Lee and Hwangyu Lim and Sejoon Lim},
      year={2022},
      eprint={2207.10003},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

