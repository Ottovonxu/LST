# LST
PyTorch Implementation for Locality Sensitive Teaching


Required Package:

    pip install torch==1.10
    pip install scipy 
    pip install cupy-cuda102 
    pip install pynvrtc 
    pip install Cython 

Before Running: Open terminal at this folder

    cd lsh-imt
    make
    python setup.py install
    cd ..

For running:

    python lst.py
    
Citation:

    @article{xu2021locality,
      title={Locality sensitive teaching},
      author={Xu, Zhaozhuo and Chen, Beidi and Li, Chaojian and Liu, Weiyang and Song, Le and Lin, Yingyan and Shrivastava, Anshumali},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
