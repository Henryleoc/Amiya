# Amiya
The fine-tuning code in this version is based on the modifications made to a Python script file named “v0171_fine_gork_masked_corr_huber_loss_minmax_norm_r3.py”. 

The main changes are as follows: 

1. A classifier has been added to the EnhancedBipartiteTransform class.
2. The _compute_corr class has been modified; it now uses MCC for met-wise calculations and cosine-similarity for spot-wise calculations.
3. The _compute_loss class has been updated, changing the Huber loss function to Focal Loss + KL divergence, where the weight of Focal Loss is calculated based on per-met weights. Although there have been no changes to the expression of total_loss, lw_corr is set to 0.0 during the current training phase with the aim of allowing the model to first learn classification.
4. Based on these adjustments, the _update_metrics class has been revised to include Recall and Precision metrics for monitoring classification performance.
