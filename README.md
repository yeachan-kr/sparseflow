# SparseFlow (ACl 2024)
SparseFlow: Accelerating Transformers by Sparsifying Information Flows (ACL 2024)

### Training SparseFlow

**Step1: Training base models for warm-up**
~~~
python -u ../main.py  \
    --dataset 'sst2' \
    --model 'bert-large-uncased' \
    --alg 'full' \
    --batch_size [batch size] \
    --max_seq_len [max sequence] \
    --epochs [warmup epoch] \
    --lr 2e-5 \
    --init_seed 42 \
~~~


**Step2: Train SparseFlow with consistency loss**

~~~
python -u ../main.py  \
    --dataset 'sst2' \
    --model 'bert-large-uncased' \
    --teacher_model [warm-up model name] \
    --alg 'popu' \
    --reg_weight [regularization weight] \
    --n_moe [number of mixture of experts in sparseflow] \
    --p_decay [target sparsity] \
    --batch_size [batch size] \
    --max_seq_len [max sequence] \
    --epochs [training epoch] \
    --lr 2e-5 \
    --init_seed 42
~~~


### Citation

```bibtex
@inproceedings{kim-lee-2024-sparseflow,
    title = "{S}parse{F}low: Accelerating Transformers by Sparsifying Information Flows",
    author = "Kim, Yeachan  and
      Lee, SangKeun",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.323",
    pages = "5937--5948",
}

```
