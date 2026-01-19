import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def conf_matrix(scores_negatives, scores_positives, threshold):
    conf_mat=np.zeros((2,2))
    conf_mat[0,0]=np.sum(scores_negatives<=threshold)
    conf_mat[0,1]=np.sum(scores_negatives>threshold)
    conf_mat[1,0]=np.sum(scores_positives<=threshold)
    conf_mat[1,1]=np.sum(scores_positives>threshold)
    return conf_mat

def tpr_fpr(scores_negatives, scores_positives, threshold):
    tpr=np.sum(scores_positives>threshold)/(np.sum(scores_positives>threshold)+np.sum(scores_negatives>threshold))
    fpr=np.sum(scores_negatives>threshold)/(np.sum(scores_positives>threshold)+np.sum(scores_negatives>threshold))
    return (tpr,fpr)

def accuracy(scores_negatives, scores_positives, threshold):
    return (np.sum(scores_positives>threshold)+np.sum(scores_negatives<=threshold))/(np.sum(scores_positives>threshold)+np.sum(scores_negatives<=threshold)+np.sum(scores_negatives>threshold)+np.sum(scores_positives<=threshold))

def precision_recall(scores_negatives, scores_positives, threshold):
    precision=np.sum(scores_positives>threshold)/(np.sum(scores_positives>threshold)+np.sum(scores_positives<=threshold))
    recall=np.sum(scores_positives>threshold)/(np.sum(scores_positives>threshold)+np.sum(scores_negatives<=threshold))
    return (precision,recall)

def f_beta(scores_negatives, scores_positives, threshold, beta):
    precision,recall=precision_recall(scores_negatives, scores_positives, threshold)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def compute_logits(dataloader, model, device):
    model.eval()
    all_logits = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
    return torch.cat(all_logits, dim=0)


def roc_auc(scores_negatives, scores_positives):
    # Combine scores and create labels
    scores = np.concatenate((scores_negatives, scores_positives))
    labels = np.concatenate((np.zeros(len(scores_negatives)), np.ones(len(scores_positives))))

    # Sort scores and labels
    sorted_indices = np.argsort(scores)
    scores = scores[sorted_indices]
    labels = labels[sorted_indices]

    # Initialize TPR and FPR
    tpr = []
    fpr = []
    n_pos = np.sum(labels)
    n_neg = len(labels) - n_pos

    tp = n_pos
    fp = n_neg

    # Compute TPR and FPR at each threshold
    for i in range(len(scores)):
        if labels[i] == 1:  # True positive
            tp -= 1
        else:  # False positive
            fp -= 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    # Compute AUROC (Area Under the Curve)
    auroc = - np.trapezoid(tpr, fpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    return auroc

def compute_threshold(scores, target_tpr=0.95):
    sorted_scores = np.sort(scores)
    target_index = int(np.ceil((1-target_tpr) * len(sorted_scores))) - 1

    # Handle edge cases
    target_index = max(0, target_index)  # Ensure index is non-negative
    target_index = min(len(sorted_scores) - 1, target_index)  # Ensure index is within bounds

    # Select the threshold
    threshold = sorted_scores[target_index]

    return threshold

# MLS Score
def mls(logits):
    scores = - torch.max(logits, dim=1)[0]
    return scores.cpu().numpy()

# MSP Score
def msp(logits):
    probas = torch.softmax(logits, dim=1)
    max_probas_scores = - torch.max(probas, dim=1)[0]
    return max_probas_scores.cpu().numpy()

# Energy Score
def energy(logits, temp=10):
    energies = - temp * torch.logsumexp(logits / temp, dim=1)
    return energies.cpu().numpy()

# Entropy Score
def entropy(logits):
    probas = torch.softmax(logits, dim=1)
    entropies = - torch.sum(probas * torch.log(probas + 1e-8), dim=1)
    return entropies.cpu().numpy()


def plot_scores(scores_positives,scores_negatives,method,writer):
    plt.figure(figsize=(10, 6))
    plt.hist(scores_negatives, bins=50, alpha=0.5, label='Negative Samples')
    plt.hist(scores_positives, bins=50, alpha=0.5, label='Positive Samples')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {method} Scores')
    plt.legend()
    writer.add_figure(f'Anomaly_Scores/{method}', plt.gcf())
    plt.show()   

class Mahalanobis():
    def __init__(self):
        self.mus = None
        self.inv_cov = None
        self.labels = None

    def fit(self, features, labels):
        self.labels = np.unique(labels)
        self.mus = {}
        covs = {}
        for label in self.labels:
            label_mask  = (labels == label)
            label_features = features[label_mask]
            self.mus[label.item()] = label_features.mean(dim=0)
            covs[label.item()] = torch.cov(label_features.T) * label_features.size(0)

        cov = sum(covs.values()) / features.size(0)
        self.inv_cov = torch.linalg.pinv(cov)

    def _mahalanobis_distance(self, x, mu, inv_cov):
        diff = x - mu
        return diff @ inv_cov @ diff.T

    def compute_scores(self, test_features):
        scores = []
        for test_feature in test_features:
            distances = torch.tensor([
                self._mahalanobis_distance(test_feature, self.mus[label.item()], self.inv_cov)
                for label in self.labels
            ])
            scores.append(torch.min(distances))
        return torch.stack(scores).cpu().numpy()
    

    

class DKNN:
    def __init__(self, k=50, batch_size=256):
        self.k = k
        self.batch_size = batch_size
        self.fit_features = None

    def _l2_normalization(self, feat):
        norms = torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-10  # Avoid division by zero
        return feat / norms

    def fit(self, fit_dataset):
        self.fit_features = self._l2_normalization(fit_dataset)

    def compute_scores(self, test_features):
        test_features = self._l2_normalization(test_features)
        scores = []

        # Process test features in batches
        for i in range(0, test_features.size(0), self.batch_size):
            batch = test_features[i:i + self.batch_size]
            # Compute pairwise distances for the batch
            distances = torch.cdist(batch, self.fit_features, p=2)  # (batch_size, num_fit_samples)
            # Sort distances and extract the k-th nearest
            sorted_distances, _ = torch.sort(distances, dim=1)
            scores.append(sorted_distances[:, self.k - 1])  # k-th nearest distance

        # Concatenate scores from all batches
        return torch.cat(scores, dim=0).cpu().numpy()
    

def compute_threshold(scores, target_tpr=0.95):
    sorted_scores = np.sort(scores)
    target_index = int(np.ceil((1-target_tpr) * len(sorted_scores))) - 1

    # Handle edge cases
    target_index = max(0, target_index)  # Ensure index is non-negative
    target_index = min(len(sorted_scores) - 1, target_index)  # Ensure index is within bounds

    # Select the threshold
    threshold = sorted_scores[target_index]

    return threshold

def compute_features(dataset, model, device):
    all_features = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image, _ = dataset[i]  # Get each image directly from the dataset
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            features = model(image, return_features=True)
            all_features.append(features)
    return torch.cat(all_features, dim=0)  # Concatenate all logits into a single tensor

def plot_confusion_matrix(model, dataloader,criterion, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    accuracy =  correct / total
    avg_test_loss = test_loss / len(dataloader)

    
    
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {100*accuracy:.2f}%")
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return (cm,accuracy)