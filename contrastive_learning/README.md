# Tutorial

Follow this github to implement the SupCon and SimCLR models: https://github.com/HobbitLong/SupContrast?tab=readme-ov-file

## Dataset Structure
It is important to note that all the datasets follow this folder structure
```
root_dir/
    class_1/
        image1.jpg
        image2.jpg
        ...
    class_2/
        image1.jpg
        image2.jpg
        ...
```

## Supervised Contrastive Learning (SupCon)
### Stage 1 (Encoder Training)
```
python main_supcon.py --batch_size 64 \
  --learning_rate 0.05 \
  --model resnet18 \
  --epochs 200 \
  --size 112 \
  --temp 0.1 \
  --cosine \
  --dataset path \
  --data_folder ../data/cropped_data/train_val \
  --mean "(0.4914, 0.4822, 0.4465)" \
  --std "(0.2675, 0.2565, 0.2761)"
```

### Stage 2 (Classification)
```
python main_linear.py --batch_size 64 \
  --learning_rate 0.005 \
  --size 112 \
  --epochs 30 \
  --model resnet18 \
  --ckpt save/SupCon/path_models/SupCon_path_resnet18_lr_0.05_decay_0.0001_bsz_64_temp_0.1_size_112_scale02_1_cosine/last.pth \
  --dataset path \
  --data_folder ../data/cropped_data/ \
  --mean "(0.4914, 0.4822, 0.4465)" \
  --std "(0.2675, 0.2565, 0.2761)"
```

## SimCLR
### Stage 1 (Encoder Training)
The major CLI difference between SupCon and SimCLR is that SimCLR has '--method SimCLR'
```
python main_supcon.py --batch_size 64 \
  --learning_rate 0.05 \
  --model resnet18 \
  --epochs 200 \
  --size 112 \
  --temp 0.5 \
  --cosine \
  --dataset path \
  --data_folder ../data/cropped_data/train_val \
  --mean "(0.4914, 0.4822, 0.4465)" \
  --std "(0.2675, 0.2565, 0.2761)" \
  --method SimCLR
```
### Stage 2 (Classification)
```
python main_linear.py --batch_size 64 \
  --learning_rate 0.05 \
  --size 112 \
  --epochs 30 \
  --model resnet18 \
  --ckpt save/SupCon/path_models/SimCLR_path_resnet18_lr_0.05_decay_0.0001_bsz_64_temp_0.5_size_112_scale02_1_cosine/last.pth \
  --dataset path \
  --data_folder ../data/cropped_data/ \
  --mean "(0.4914, 0.4822, 0.4465)" \
  --std "(0.2675, 0.2565, 0.2761)"
```
