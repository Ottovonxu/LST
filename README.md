# Locality Sensitive Teaching

By Zhaozhuo Xu, Beidi Chen, Chaojian Li, Weiyang Liu, Le Song, Yingyan Lin, Anshumali Shrivastava

## Introduction

The emergence of the Internet-of-Things (IoT) sheds light on applying the machine teaching (MT) algorithms for online personalized education on home devices. This direction becomes more promising during the COVID-19 pandemic when in-person education becomes infeasible. However, as one of the most influential and practical MT paradigms, iterative machine teaching (IMT) is prohibited on IoT devices due to its inefficient and unscalable algorithms. IMT is a paradigm where a teacher feeds examples iteratively and intelligently based on the learner's status. In each iteration, current IMT algorithms greedily traverse the whole training set to find an example for the learner, which is computationally expensive in practice.  We propose a novel teaching framework, Locality Sensitive Teaching (LST), based on locality sensitive sampling, to overcome these challenges. LST has provable near-constant time complexity, which is exponentially better than the existing baseline. With at most 425.12x speedups and 99.76% energy savings over IMT, LST is the first algorithm that enables energy and time efficient machine teaching on IoT devices. Owing to LST's substantial efficiency and scalability, it is readily applicable in real-world education scenarios.

We provide a PyTorch Implementation for Locality Sensitive Teaching, and the paper is available at [OpenReview](https://openreview.net/forum?id=Rav_oC35ToB).

## Get Started

Required Package:

    pip install torch==1.10
    pip install scipy 
    pip install cupy-cuda102 
    pip install pynvrtc 
    pip install Cython 

LSH Installation:

    cd lsh_imt
    make
    python setup.py install
    cd ..

For running:

    python lst.py


## Citation

If you use this codebase, or otherwise found our work valuable, please cite:


    @inproceedings{xu2021locality,
      title={Locality sensitive teaching},
      author={Xu, Zhaozhuo and Chen, Beidi and Li, Chaojian and Liu, Weiyang and Song, Le and Lin, Yingyan and Shrivastava, Anshumali},
      booktitle={NeurIPS},
      year={2021}
    }
