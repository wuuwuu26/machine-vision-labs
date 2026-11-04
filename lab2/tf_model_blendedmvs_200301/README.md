# BlendedMVS checkpoint (not tracked)

This folder is a placeholder for the MVSNet TensorFlow checkpoint used in Lab 2.

Expected layout:

```text
lab2/tf_model_blendedmvs_200301/
└── tf_model_blendedmvs/
    ├── 3DCNNs/
    │   └── model.ckpt-150000.{data-00000-of-00001,index,meta}
    └── GRU/
        └── model.ckpt-150000.{data-00000-of-00001,index,meta}
```

The checkpoint is available from the official MVSNet repository:

<https://github.com/YoYo000/MVSNet>

The 3DCNNs branch is the one used by `run_stage1.sh` (`--regularization 3DCNNs`, `--ckpt_step 150000`). The checkpoint is excluded from git because of its size (~120 MB).
