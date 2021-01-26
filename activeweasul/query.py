import logging
import numpy as np
import random
from scipy.stats import entropy


class ActiveLearningQuery():
    """Sample a data point from unlabeled pool following an active learning strategy, based on probabilistic label output.

    Possible strategies include "maxkl", "margin" and "nashaat".
    """

    def __init__(self, query_strategy):
        self.query_strategy = query_strategy
        strategy_mapping = {"margin": self.margin_strategy,
                            "maxkl": self.maxkl_strategy,
                            "nashaat": self.nashaat_strategy}
        try:
            self.query = strategy_mapping[query_strategy]
        except KeyError:
            logging.warning("Provided active learning strategy not valid, setting to maxkl")
            self.query = self.maxkl_strategy

    def margin(self, probs):
        """P(Y=1|...) - P(Y=0|...)"""

        abs_diff = np.abs(probs[:, 1] - probs[:, 0])

        return abs_diff

    def margin_strategy(self):
        """Choose bucket to sample from based on uncertainty of probabilistic label

        Returns:
            list: List of indices of buckets to choose from
        """

        # Find bucket uncertainties
        bucket_margins = self.margin(self.bucket_probs)
        self.bucket_values = bucket_margins

        # Select buckets with highest uncertainty
        return np.where(bucket_margins == np.min(bucket_margins[self.is_valid_bucket]))[0]

    def nashaat_strategy(self):
        """Query strategy by Nashaat et al.

        Returns:
            list: List of indices of buckets to choose from
        """

        # Disagreement factor, buckets with labeling function conflicts
        has_conflicts = ~((self.unique_combs.sum(axis=1) == 0) | (self.unique_combs.sum(axis=1) == self.unique_combs.shape[1]))
        
        bucket_margins = self.margin(self.bucket_probs)

        # Select buckets with highest uncertainty and disagreeing weak labels
        return np.where(bucket_margins == np.min(bucket_margins[self.is_valid_bucket & has_conflicts]))[0]

    def maxkl_strategy(self):
        """Choose bucket of points to sample from following MaxKL query strategy

        Returns:
            list: List of indices of buckets to choose from
        """
        # TODO: rewrite to update only updated distribution from sampled bucket
        # Instead of computing everything again every iteration

        # Label model distributions
        lm_posteriors = self.bucket_probs.clip(1e-5, 1-1e-5)

        # Sample distributions
        # D_KL(LM distribution||Sample distribution)
        rel_entropy = np.zeros(len(lm_posteriors))
        sample_posteriors = np.zeros(lm_posteriors.shape)
        # Iterate over buckets
        for i in range(len(lm_posteriors)):
            # Collect points in bucket
            bucket_items = self.ground_truth_labels[np.where(self.unique_inverse == i)[0]]
            # Collect labeled points in bucket
            bucket_gt = list(bucket_items[bucket_items != -1])
            # Add initial labeled point
            if not bucket_gt:
                bucket_gt.append(int(np.round(self.probs["bucket_labels_train"][0][i].clip(0, 1))))
            bucket_gt = np.array(bucket_gt)

            # Bucket distribution, clip to avoid D_KL undefined
            eps = 1e-2/(len(bucket_gt))
            sample_posteriors[i, 1] = bucket_gt.mean().clip(eps, 1-eps)
            sample_posteriors[i, 0] = 1 - sample_posteriors[i, 1]

            # KL divergence
            rel_entropy[i] = entropy(lm_posteriors[i, :], sample_posteriors[i, :])#/len(bucket_gt)

        self.bucket_values = rel_entropy

        # Select buckets with highest KL divergence
        return np.where(rel_entropy == np.max(rel_entropy[self.is_valid_bucket]))[0]

    def sample(self, probs):
        """Choose data point to label following query strategy

        Args:
            probs (numpy.array): Array with probabilistic labels for training dataset

        Returns:
            int: Index of chosen point
        """
        all_abstain = (self.label_matrix == -1).sum(axis=1) == self.label_matrix.shape[1]
        self.is_in_pool = (self.ground_truth_labels == -1) & ~all_abstain & (self.y_train != -1)
        self.valid_buckets = np.unique(self.unique_inverse[self.is_in_pool])
        self.is_valid_bucket = np.array([True if i in self.valid_buckets else False for i in range(len(self.unique_idx))])
        self.bucket_probs = probs.detach().numpy()[self.unique_idx]

        pick = random.uniform(0, 1)
        if pick < self.randomness:
            # Choose random bucket instead of following a specific query strategy
            chosen_bucket = np.random.choice(self.valid_buckets)
        else:
            chosen_bucket = np.random.choice(self.query())

        return random.choice(np.where((self.unique_inverse == chosen_bucket) & self.is_in_pool)[0])
