# NAS-Bench-301

This repository containts code for the paper "NAS-Bench-301 and the Case for Surrogate Benchmarks for Neural Architecture Search".

The surrogate models for v1.0 can be downloaded on [figshare](https://figshare.com/articles/software/nasbench301_models_v1_0_zip/13061837). We also release the [full dataset](https://figshare.com/articles/dataset/nasbench301_full_data/13285934) including training logs, final performances and other metrics such as model parameters (a [lightweight version](https://figshare.com/articles/dataset/nasbench301_data/13247021) without training logs is also available).

To install all requirements (this may take a few minutes), run

```sh
$ cat requirements.txt | xargs -n 1 -L 1 pip install
$ pip install torch-scatter==2.0.4+cu102 torch-sparse==0.6.3+cu102 torch-cluster==1.5.5+cu102 torch-spline-conv==1.2.0+cu102 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
$ pip install torch-geometric
```

To run a quick example, adapt the model paths in 'nasbench301/example.py' and from the base directory run

```sh
$ export PYTHONPATH=$PWD
$ python3 nasbench301/example.py
```
