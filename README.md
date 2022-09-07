# Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation

This is the Pytorch implementation of STAR-HiT for next Point-of-Interest Recommendation.

## Introduction

Spatio-Temporal context AggRegated Hierarchical Transformer (STAR-HiT)  is a Transformer-based model for next POI recommendation. STAR-HiT employs stacked hierarchical encoders to recursively encode the spatio-temporal context and explicitly locate subsequences of different granularities, so as to capture the latent hierarchical structure of user movements.

## Environmemnt Requirements

* python == 3.6.13
* pytorch == 1.2.0
* numpy == 1.19.2
* pandas == 1.1.5
* sklearn == 0.24.2
* joblib == 1.0.1

## Run the Codes

1. Train: replace the `data_root` in `utils/parser.py` to where you store the dataset, then run the `train.py` to train the model. The trained model will be saved in `models/`.

2. Test: replace the `trained_model_path` and `model_name` in `predict.py` to point to your trained model, then run the `predict.py` to test the model. The test results will be saved to your `trained_model_path`.

3. We also provide a model in `models/` that is trained on Foursquare NYC. Download the processed datasets below and direct run the `predict.py` to see the test results.

## Datasets

We provide a processed datasets [Foursquare NYC](https://drive.google.com/drive/folders/1W1emflA4aMKrtStxTSJLqbb_HjhGkAS-?usp=sharing). The details are described as follows:

The `data_info.csv` contains the metadata of the datases, including the number of users, the number of POIs, the max length of check-in sequences.

The `{}.pkl` files (both of three phases) contains the data for training, validation and test, composed of a set of dictionary as:

```
{
    "user_id": user id,
    "poi_id_seq_in": check-in poi ids of the input sequence,
    "poi_dist_mat_in": check-in poi distance matrix,
    "poi_timediff_mat_in": check-in time interval matrix (level),
    "poi_out": the target check-in poi id,
    "poi_distance": check-in poi distance, # vector, baselines used only
    "poi_time_interval": check-in time interval (timestamp), # vector, baselines used only
    "poi_timestamp": poi check-in timestamp, # vector, including the target, baselines used only
    "poi_location_quadtree": poi_location_quadtree,  # vector, including the target, baselines used only
    "poi_loc_quad_idx": poi_loc_quad_idx, # including the target, baselines used only
    "poi_dtime_mat_in": time interal matrix (timestamp), # baselines used only
    "poi_seq_out": the target check-in poi sequence # baselines used only
}
```

## Citation

If you find our codes and datasets helpful, please kindly cite the following papers:

```
@article{STARHiT,
  author    = {Jiayi Xie and
               Zhenzhong Chen},
  title     = {Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation},
  journal   = {CoRR},
  volume    = {abs/2209.01559},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2209.01559},
  doi       = {arXiv:2209.01559},
  eprinttype = {arXiv},
  eprint    = {2209.01559},
}
```

or
```
@Eprint{STARHiT,
  author    = {Jiayi Xie and
               Zhenzhong Chen},
  title     = {Hierarchical Transformer with Spatio-Temporal Context Aggregation for Next Point-of-Interest Recommendation},
  year      = {2016},
  archivePrefix = {arXiv},
  eprint    = {2209.01559},
}
```