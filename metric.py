import numpy as np
from scipy.stats import pearsonr

def correlation_dissimilarity(emb1, emb2):
    """
    emb1 (np.array) : embedding in one feature space
    emb2 (np.array) : embedding in another feature space
    """
    dissim1 = 1. - np.corrcoef(emb1)
    dissim2 = 1. - np.corrcoef(emb2)

    triu_indices = np.triu_indices_from(dissim1, k=1)
    flat1 = dissim1[triu_indices]
    flat2 = dissim2[triu_indices]

    # Compute second-order similarity (Pearson correlation)
    r, _ = pearsonr(flat1, flat2)
    return r


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_linear_classifier(X, y, test_size=0.2, random_state=42, **kwargs):
    """
    Trains a linear classifier (Logistic Regression) and returns the model and accuracy.

    Parameters:
    X (array-like): Feature matrix
    y (array-like): Target vector
    test_size (float): Proportion of data to use for testing (default: 0.2)
    random_state (int): Random seed for reproducibility (default: 42)
    **kwargs: Additional arguments to pass to LogisticRegression

    Returns:
    tuple: (trained_model, accuracy_score)
    """
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Initialize and train the linear classifier
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def encode_set(encoder_function: callable, loader, original_loader, device="cpu"):
    all_embeddings = []
    all_labels = []
    all_original_images = []

    with torch.no_grad():
        for (images_dino, targets), (images_orig, _) in zip(loader, original_loader):
            images_dino = images_dino.to(device, non_blocking=True)
            embeddings = encoder_function(images_dino).cpu()
            all_embeddings.append(embeddings)
            all_labels.append(targets)
            all_original_images.append(images_orig)

    embeddings = torch.cat(all_embeddings)  # [N, D]
    labels = torch.cat(all_labels)
    original_images = torch.cat(all_original_images)
    original_images = original_images.reshape(original_images.shape[0], -1)

    return (embeddings.numpy(),
            labels.numpy(),
            original_images.numpy())


def run(encoder_function: callable, loader: torch.utils.data.DataLoader, original_loader: torch.utils.data.DataLoader,
        logger = None, device = "cpu"):
    embeddings_np, labels_np, original_images_np = encode_set(encoder_function, loader, original_loader, device)
    if not logger is None:
        logger.log({
            "classification_accuracy" : train_linear_classifier(embeddings_np, labels_np)[1],
            "second_order_similarity" : correlation_dissimilarity(embeddings_np, original_images_np)
        })
    return embeddings_np, labels_np, original_images_np
