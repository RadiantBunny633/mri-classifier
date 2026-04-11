"""
RF — MRI Alzheimer's Classifier (Built from Scratch)
===========================================================
Custom Decision Tree and Random Forest implementation.
"""

import numpy as np


IMAGE_SIZE = 128


def flatten_images(images: np.ndarray) -> np.ndarray:
    """
    Convert (N, H, W) image array to (N, H*W) feature matrix.
    """
    return images.reshape(len(images), -1)


# ============== DECISION TREE ==============

def gini_impurity(labels):
    """
    Calculate Gini impurity for a set of labels.
    Gini = 1 - sum(p_i^2) where p_i is the proportion of class i.
    """
    if len(labels) == 0:
        return 0.0
    classes, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(labels)
    return 1.0 - np.sum(proportions ** 2)


def information_gain(left_labels, right_labels, parent_gini):
    """
    Calculate information gain from a split.
    IG = parent_gini - weighted_avg(child_gini)
    """
    n = len(left_labels) + len(right_labels)
    if n == 0:
        return 0.0
    weight_left = len(left_labels) / n
    weight_right = len(right_labels) / n
    child_gini = weight_left * gini_impurity(left_labels) + weight_right * gini_impurity(right_labels)
    return parent_gini - child_gini


def find_best_split(X, y, feature_indices):
    """
    Find the best feature and threshold to split on.
    Only considers features in feature_indices (for random subsets).
    """
    best_gain = -1
    best_feature = None
    best_threshold = None

    parent_gini = gini_impurity(y)

    for feature_idx in feature_indices:
        values = X[:, feature_idx]
        # Use a subset of thresholds for efficiency
        unique_vals = np.unique(values)
        if len(unique_vals) <= 20:
            thresholds = unique_vals
        else:
            thresholds = np.percentile(values, np.linspace(10, 90, 20))

        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            gain = information_gain(y[left_mask], y[right_mask], parent_gini)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


class DecisionTreeNode:
    """A single node in the decision tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature          # feature index to split on
        self.threshold = threshold      # threshold value for split
        self.left = left                # left child (<=)
        self.right = right              # right child (>)
        self.prediction = prediction    # class prediction (leaf nodes only)


class DecisionTree:
    """
    Decision Tree classifier built from scratch using Gini impurity.
    """
    def __init__(self, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features="sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
        self.n_features = None

    def _get_n_features_to_sample(self, total_features):
        """Determine how many features to consider at each split."""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(total_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(total_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, total_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * total_features))
        else:
            return total_features

    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth or
            n_classes == 1 or
            n_samples < self.min_samples_split):
            # Make leaf node with majority class
            classes, counts = np.unique(y, return_counts=True)
            prediction = classes[np.argmax(counts)]
            return DecisionTreeNode(prediction=prediction)

        # Randomly select features to consider
        n_features_to_sample = self._get_n_features_to_sample(self.n_features)
        feature_indices = np.random.choice(self.n_features, size=n_features_to_sample, replace=False)

        # Find best split
        best_feature, best_threshold, best_gain = find_best_split(X, y, feature_indices)

        # If no good split found, make leaf
        if best_gain <= 0 or best_feature is None:
            classes, counts = np.unique(y, return_counts=True)
            prediction = classes[np.argmax(counts)]
            return DecisionTreeNode(prediction=prediction)

        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Check min_samples_leaf
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            classes, counts = np.unique(y, return_counts=True)
            prediction = classes[np.argmax(counts)]
            return DecisionTreeNode(prediction=prediction)

        # Build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return DecisionTreeNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child
        )

    def fit(self, X, y):
        """Train the decision tree."""
        self.n_features = X.shape[1]
        self.root = self._build_tree(X, y)
        return self

    def _predict_single(self, x, node):
        """Predict class for a single sample."""
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X):
        """Predict classes for all samples."""
        return np.array([self._predict_single(x, self.root) for x in X])


# ============== RANDOM FOREST ==============

class RandomForestClassifier:
    """
    Random Forest classifier built from scratch.
    Uses bootstrap aggregation (bagging) with random feature subsets.
    """
    def __init__(self, n_estimators=100, max_depth=15, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt", class_weight=None,
                 random_state=42, verbose=0, n_jobs=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs  # not used but kept for compatibility
        self.trees = []
        self.classes_ = None
        self.feature_importances_ = None

    def _bootstrap_sample(self, X, y):
        """Create a bootstrap sample (sample with replacement)."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)

        if self.class_weight == "balanced":
            # Adjust sampling to balance classes
            classes, counts = np.unique(y, return_counts=True)
            max_count = counts.max()
            balanced_indices = []
            for cls, count in zip(classes, counts):
                cls_indices = np.where(y == cls)[0]
                # Oversample minority classes
                sampled = np.random.choice(cls_indices, size=max_count, replace=True)
                balanced_indices.extend(sampled)
            indices = np.array(balanced_indices)
            np.random.shuffle(indices)

        return X[indices], y[indices]

    def fit(self, X, y):
        """Train the random forest."""
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.trees = []

        if self.verbose:
            print(f"Training Random Forest with {self.n_estimators} trees...")

        for i in range(self.n_estimators):
            if self.verbose and (i + 1) % 25 == 0:
                print(f"  Tree {i + 1}/{self.n_estimators} complete")

            # Bootstrap sample
            X_boot, y_boot = self._bootstrap_sample(X, y)

            # Build tree with random feature subsets
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        if self.verbose:
            print("Training complete.")

        # Compute feature importances (approximate using prediction variance)
        self._compute_feature_importances(X, y)

        return self

    def _compute_feature_importances(self, X, y):
        """
        Approximate feature importances using permutation importance
        on a subset of the data for efficiency.
        """
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)

        # Use a subset for efficiency
        n_subset = min(500, len(X))
        indices = np.random.choice(len(X), size=n_subset, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]

        base_accuracy = (self.predict(X_sub) == y_sub).mean()

        # Permute each feature and measure drop in accuracy
        # For efficiency with 16k features, sample a subset of features
        n_features_to_check = min(500, n_features)
        feature_sample = np.random.choice(n_features, size=n_features_to_check, replace=False)

        for feat_idx in feature_sample:
            X_permuted = X_sub.copy()
            X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
            permuted_accuracy = (self.predict(X_permuted) == y_sub).mean()
            importance = base_accuracy - permuted_accuracy
            self.feature_importances_[feat_idx] = max(0, importance)

        # Normalize
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def predict(self, X):
        """Predict classes using majority voting across all trees."""
        # Collect predictions from all trees
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote for each sample
        predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            votes = all_predictions[:, i]
            classes, counts = np.unique(votes, return_counts=True)
            predictions[i] = classes[np.argmax(counts)]

        return predictions

    def predict_proba(self, X):
        """Predict class probabilities using vote proportions."""
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        probabilities = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            votes = all_predictions[:, i]
            for j, cls in enumerate(self.classes_):
                probabilities[i, j] = (votes == cls).sum() / self.n_estimators

        return probabilities


def build_model() -> RandomForestClassifier:
    """
    Construct the Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        verbose=1,
    )


def train_model(rf, X_train, y_train):
    """
    Fit the RF on flattened pixel vectors.
    """
    print(f"Training Random Forest...")
    print(f"  Samples:  {X_train.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    rf.fit(X_train, y_train)
    return rf


if __name__ == "__main__":
    X_dummy = np.random.rand(20, IMAGE_SIZE * IMAGE_SIZE).astype(np.float32)
    y_dummy = np.random.randint(0, 4, size=20)

    rf = build_model()
    rf = train_model(rf, X_dummy, y_dummy)
    preds = rf.predict(X_dummy)

    print(f"\nSmoke test passed.")
    print(f"Input shape:  {X_dummy.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Classes seen: {np.unique(preds)}")