import numpy as np
import time

# =============================================================
#  LEITURA DE DADOS
# =============================================================

def parse_arff(filepath, target_name=None, target_index=-1, drop_names=None):
    """
    LÃª arquivos no formato ARFF (usado pelo OpenML / Weka).
    Retorna X (numpy array de features) e y (numpy array de rÃ³tulos).
    """
    drop_names = set(drop_names or [])
    attributes = []
    data_lines = []
    in_data = False

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            lower = line.lower()
            if lower.startswith('@attribute'):
                rest = line[10:].strip()
                if rest.startswith("'"):
                    end = rest.index("'", 1)
                    name = rest[1:end]
                    typ = rest[end+1:].strip()
                else:
                    parts = rest.split(None, 1)
                    name = parts[0]
                    typ = parts[1] if len(parts) > 1 else 'numeric'
                attributes.append((name, typ.lower()))
            elif lower.startswith('@data'):
                in_data = True
            elif in_data and line:
                data_lines.append(line)

    if not attributes:
        raise ValueError(f"Nenhum atributo encontrado em {filepath}")

    n_attrs = len(attributes)
    attr_names = [name for name, _ in attributes]
    if target_name is not None:
        if target_name not in attr_names:
            raise ValueError(f"Atributo alvo '{target_name}' nao encontrado em {filepath}")
        target_idx = attr_names.index(target_name)
    else:
        target_idx = target_index if target_index >= 0 else n_attrs + target_index

    feature_indices = [
        i for i, name in enumerate(attr_names)
        if i != target_idx and name not in drop_names
    ]

    X, y = [], []
    for line in data_lines:
        if line.startswith('{'):
            # Formato esparso - pula
            continue
        values = [v.strip() for v in line.split(',')]
        if len(values) < n_attrs:
            continue
        feats = []
        ok = True
        for i in feature_indices:
            try:
                feats.append(float(values[i]))
            except ValueError:
                ok = False
                break
        if ok:
            X.append(feats)
            y.append(values[target_idx].strip().strip("'\""))

    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    if X.size == 0:
        raise ValueError(f"Nenhuma linha valida foi carregada de {filepath}")

    # Remove linhas com NaN
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask], [attr_names[i] for i in feature_indices]


def parse_csv(filepath, target_is_last=True, has_header=True):
    """
    LÃª arquivos CSV para regressÃ£o.
    Retorna X (features) e y (alvo numÃ©rico).
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]

    start = 1 if has_header else 0
    # Tenta detectar automaticamente se tem cabeÃ§alho
    try:
        float(lines[0].split(',')[0])
        start = 0
    except ValueError:
        start = 1

    rows = []
    for line in lines[start:]:
        vals = [v.strip() for v in line.split(',')]
        rows.append(vals)

    n_cols = len(rows[0]) if rows else 0
    X, y = [], []
    for row in rows:
        if len(row) < n_cols:
            continue
        try:
            if target_is_last:
                feats = [float(row[i]) for i in range(n_cols - 1)]
                target = float(row[-1])
            else:
                feats = [float(row[i]) for i in range(1, n_cols)]
                target = float(row[0])
            X.append(feats)
            y.append(target)
        except ValueError:
            continue

    feature_names = [f"x{i}" for i in range(len(X[0]))] if X else []
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), feature_names


# =============================================================
#  PRÃ‰-PROCESSAMENTO
# =============================================================

def normalize(X_train, X_test):
    """
    NormalizaÃ§Ã£o Min-Max usando apenas os dados de treino.
    Evita data leakage (vazamento de informaÃ§Ã£o do teste pro treino).
    """
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    rng = X_max - X_min
    rng[rng == 0] = 1  # evita divisÃ£o por zero em colunas constantes
    X_train_n = (X_train - X_min) / rng
    X_test_n  = (X_test  - X_min) / rng
    return X_train_n, X_test_n


def encode_labels(y):
    """
    Transforma rÃ³tulos textuais em inteiros.
    Ex: ['cat','dog','cat'] -> [0, 1, 0], classes=['cat','dog']
    """
    classes = np.unique(y)
    mapping = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([mapping[v] for v in y], dtype=int)
    return y_enc, classes


# =============================================================
#  VALIDAÃ‡ÃƒO CRUZADA (K-FOLD) MANUAL
# =============================================================

def kfold_indices(n, k=5, seed=42):
    """
    Gera os Ã­ndices de treino e teste para K-Fold Cross Validation.
    Retorna lista de tuplas (idx_treino, idx_teste).
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)
    splits = []
    for i in range(k):
        test_idx  = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, test_idx))
    return splits


# =============================================================
#  MÃ‰TRICAS DE CLASSIFICAÃ‡ÃƒO  (sem sklearn)
# =============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_per_class(y_true, y_pred, n_classes):
    """Calcula TP, FP, FN e TN para cada classe."""
    TP = np.zeros(n_classes)
    FP = np.zeros(n_classes)
    FN = np.zeros(n_classes)
    TN = np.zeros(n_classes)
    for c in range(n_classes):
        TP[c] = np.sum((y_pred == c) & (y_true == c))
        FP[c] = np.sum((y_pred == c) & (y_true != c))
        FN[c] = np.sum((y_pred != c) & (y_true == c))
        TN[c] = np.sum((y_pred != c) & (y_true != c))
    return TP, FP, FN, TN


def confusion_matrix(y_true, y_pred, n_classes):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[int(true_value), int(pred_value)] += 1
    return matrix


def safe_divide(numerator, denominator):
    result = np.zeros_like(numerator, dtype=np.float64)
    mask = denominator > 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def precision_macro(y_true, y_pred, n_classes):
    TP, FP, _, _ = confusion_per_class(y_true, y_pred, n_classes)
    per_class = safe_divide(TP, TP + FP)
    return per_class.mean()


def recall_macro(y_true, y_pred, n_classes):
    TP, _, FN, _ = confusion_per_class(y_true, y_pred, n_classes)
    per_class = safe_divide(TP, TP + FN)
    return per_class.mean()


def specificity_macro(y_true, y_pred, n_classes):
    _, FP, _, TN = confusion_per_class(y_true, y_pred, n_classes)
    per_class = safe_divide(TN, TN + FP)
    return per_class.mean()


def f1_macro(y_true, y_pred, n_classes):
    TP, FP, FN, _ = confusion_per_class(y_true, y_pred, n_classes)
    precision = safe_divide(TP, TP + FP)
    recall = safe_divide(TP, TP + FN)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return f1.mean()


# =============================================================
#  MÃ‰TRICAS DE REGRESSÃƒO  (sem sklearn)
# =============================================================

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def r2_adjusted(y_true, y_pred, n_features):
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    if n - n_features - 1 <= 0:
        return r2
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


# =============================================================
#  KNN  (ClassificaÃ§Ã£o e RegressÃ£o)
# =============================================================

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


def manhattan_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)


def pairwise_distances(X_train, X_batch, metric):
    if metric == 'euclidean':
        train_sq = np.sum(X_train ** 2, axis=1)
        batch_sq = np.sum(X_batch ** 2, axis=1)[:, None]
        dists_sq = batch_sq + train_sq[None, :] - 2 * (X_batch @ X_train.T)
        return np.sqrt(np.maximum(dists_sq, 0.0))
    if metric == 'manhattan':
        diff = np.abs(X_batch[:, None, :] - X_train[None, :, :])
        return np.sum(diff, axis=2)
    raise ValueError("metric deve ser 'euclidean' ou 'manhattan'")


class KNNClassifier:
    """
    K-Nearest Neighbors para classificaÃ§Ã£o.
    Prediz a classe mais comum entre os K vizinhos mais prÃ³ximos.
    """
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict_one(self, x):
        if self.metric == 'euclidean':
            dists = euclidean_distance(self.X_train, x)
        else:
            dists = manhattan_distance(self.X_train, x)
        k_idx = np.argsort(dists)[:self.k]
        neighbors = self.y_train[k_idx]
        # VotaÃ§Ã£o por maioria
        values, counts = np.unique(neighbors, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X_test):
        predictions = []
        batch_size = 256 if self.metric == 'euclidean' else 32
        for start in range(0, len(X_test), batch_size):
            X_batch = X_test[start:start + batch_size]
            dists = pairwise_distances(self.X_train, X_batch, self.metric)
            k_idx = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]
            for row in k_idx:
                neighbors = self.y_train[row]
                values, counts = np.unique(neighbors, return_counts=True)
                predictions.append(values[np.argmax(counts)])
        return np.array(predictions)


class KNNRegressor:
    """
    K-Nearest Neighbors para regressÃ£o.
    Prediz a mÃ©dia dos valores dos K vizinhos mais prÃ³ximos.
    """
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _predict_one(self, x):
        if self.metric == 'euclidean':
            dists = euclidean_distance(self.X_train, x)
        else:
            dists = manhattan_distance(self.X_train, x)
        k_idx = np.argsort(dists)[:self.k]
        return np.mean(self.y_train[k_idx])

    def predict(self, X_test):
        predictions = []
        batch_size = 256 if self.metric == 'euclidean' else 32
        for start in range(0, len(X_test), batch_size):
            X_batch = X_test[start:start + batch_size]
            dists = pairwise_distances(self.X_train, X_batch, self.metric)
            k_idx = np.argpartition(dists, self.k - 1, axis=1)[:, :self.k]
            predictions.extend(np.mean(self.y_train[k_idx], axis=1))
        return np.array(predictions)


# =============================================================
#  NAIVE BAYES UNIVARIADO
#  (baseado no cÃ³digo da professora, com predict corrigido)
# =============================================================

class NaiveBayesUni:
    """
    Naive Bayes Gaussiano Univariado.
    Assume independÃªncia entre features e usa a distribuiÃ§Ã£o normal
    1D (univariada) para estimar p(xi | classe).
    """
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)
        self._mean   = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var    = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._mean[idx, :]  = X_c.mean(axis=0)
            self._var[idx, :]   = X_c.var(axis=0)
            self._priors[idx]   = X_c.shape[0] / float(n_samples)

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var  = self._var[class_idx] + 1e-9  # suavizaÃ§Ã£o numÃ©rica
        numerator   = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_one(self, x):
        posteriors = []
        for idx in range(len(self._classes)):
            prior     = np.log(self._priors[idx])
            log_lhood = np.sum(np.log(self._pdf(idx, x) + 1e-300))
            posteriors.append(prior + log_lhood)
        return self._classes[np.argmax(posteriors)]

    def predict(self, X_test):
        return np.array([self._predict_one(x) for x in X_test])


# =============================================================
#  NAIVE BAYES MULTIVARIADO
# =============================================================

class NaiveBayesMulti:
    """
    Naive Bayes Gaussiano Multivariado.
    Usa a distribuiÃ§Ã£o normal multivariada completa, levando em conta
    as correlaÃ§Ãµes entre features atravÃ©s da matriz de covariÃ¢ncia.
    """
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self._classes = np.unique(y_train)
        n_classes = len(self._classes)
        self._mean   = []
        self._cov    = []
        self._priors = np.zeros(n_classes, dtype=np.float64)
        for idx, c in enumerate(self._classes):
            X_c = X_train[y_train == c]
            self._mean.append(X_c.mean(axis=0))
            cov = np.cov(X_c, rowvar=False)
            # RegularizaÃ§Ã£o: adiciona pequeno valor na diagonal para
            # evitar matriz singular (nÃ£o invertÃ­vel)
            cov += np.eye(n_features) * 1e-6
            self._cov.append(cov)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _log_pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        cov  = self._cov[class_idx]
        d    = len(mean)
        diff = x - mean
        try:
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0:
                return -np.inf
            cov_inv  = np.linalg.inv(cov)
            exponent = -0.5 * diff @ cov_inv @ diff
            log_norm = -0.5 * (d * np.log(2 * np.pi) + log_det)
            return log_norm + exponent
        except np.linalg.LinAlgError:
            return -np.inf

    def _predict_one(self, x):
        posteriors = [
            np.log(self._priors[i] + 1e-300) + self._log_pdf(i, x)
            for i in range(len(self._classes))
        ]
        return self._classes[np.argmax(posteriors)]

    def predict(self, X_test):
        return np.array([self._predict_one(x) for x in X_test])


# =============================================================
#  REGRESSÃƒO LINEAR MÃšLTIPLA
# =============================================================

class LinearRegression:
    """
    RegressÃ£o Linear MÃºltipla pela EquaÃ§Ã£o Normal:
        Î¸ = (Xáµ€ X)â»Â¹ Xáµ€ y
    Adiciona coluna de 1s para o intercepto (bias).
    """
    def fit(self, X_train, y_train):
        n = X_train.shape[0]
        # Adiciona coluna de 1s (intercepto)
        X_b = np.hstack([np.ones((n, 1)), X_train])
        self._theta = np.linalg.pinv(X_b) @ y_train

    def predict(self, X_test):
        n = X_test.shape[0]
        X_b = np.hstack([np.ones((n, 1)), X_test])
        return X_b @ self._theta


class PerceptronSimples:
    """
    Perceptron Simples implementado manualmente.
    Para duas classes, usa a regra classica com alvos -1 e +1.
    """
    def __init__(self, learning_rate=0.01, epochs=100, shuffle=True, seed=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.shuffle = shuffle
        self.seed = seed

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        rng = np.random.default_rng(self.seed)

        if n_classes == 2:
            self.is_binary = True
            self.weights = np.zeros(n_features)
            self.bias = 0.0
            y_binary = np.where(y == self.classes[1], 1, -1)

            for _ in range(self.epochs):
                indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
                for idx in indices:
                    x_i = X[idx]
                    linear_output = np.dot(x_i, self.weights) + self.bias
                    y_predicted = 1 if linear_output >= 0 else -1
                    if y_binary[idx] != y_predicted:
                        update = self.learning_rate * y_binary[idx]
                        self.weights += update * x_i
                        self.bias += update
        else:
            self.is_binary = False
            self.weights = np.zeros((n_classes, n_features))
            self.bias = np.zeros(n_classes)
            for class_idx, class_value in enumerate(self.classes):
                y_binary = np.where(y == class_value, 1, -1)
                for _ in range(self.epochs):
                    indices = rng.permutation(n_samples) if self.shuffle else np.arange(n_samples)
                    for idx in indices:
                        x_i = X[idx]
                        linear_output = np.dot(x_i, self.weights[class_idx]) + self.bias[class_idx]
                        y_predicted = 1 if linear_output >= 0 else -1
                        if y_binary[idx] != y_predicted:
                            update = self.learning_rate * y_binary[idx]
                            self.weights[class_idx] += update * x_i
                            self.bias[class_idx] += update
        return self

    def predict(self, X):
        if self.is_binary:
            linear_output = np.dot(X, self.weights) + self.bias
            class_indices = np.where(linear_output >= 0, 1, 0)
            return self.classes[class_indices]

        linear_output = np.dot(X, self.weights.T) + self.bias
        return self.classes[np.argmax(linear_output, axis=1)]


# =============================================================
#  MULTI LAYER PERCEPTRON (REGRESSAO)
# =============================================================

class MLPRegressor:
    """
    MLP para regressao implementado manualmente com backpropagation.
    A saida e linear e o alvo e normalizado internamente para estabilizar o treino.
    """
    def __init__(self, hidden_layers=(16,), learning_rate=0.01, epochs=300,
                 activation='relu', seed=42):
        self.hidden_layers = tuple(hidden_layers)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.seed = seed
        self.weights = []
        self.biases = []

    def _activation(self, z):
        if self.activation == 'relu':
            return np.maximum(0.0, z)
        if self.activation == 'tanh':
            return np.tanh(z)
        if self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
        raise ValueError("activation deve ser 'relu', 'tanh' ou 'sigmoid'")

    def _activation_derivative(self, activated):
        if self.activation == 'relu':
            return (activated > 0).astype(np.float64)
        if self.activation == 'tanh':
            return 1.0 - activated ** 2
        if self.activation == 'sigmoid':
            return activated * (1.0 - activated)
        raise ValueError("activation deve ser 'relu', 'tanh' ou 'sigmoid'")

    def _initialize_weights(self, n_features):
        rng = np.random.default_rng(self.seed)
        layer_sizes = [n_features] + list(self.hidden_layers) + [1]
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if i == len(layer_sizes) - 2:
                limit = np.sqrt(1.0 / fan_in)
            elif self.activation == 'relu':
                limit = np.sqrt(2.0 / fan_in)
            else:
                limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.normal(0.0, limit, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        if self.y_std == 0:
            self.y_std = 1.0
        y_scaled = ((y - self.y_mean) / self.y_std).reshape(-1, 1)

        for _ in range(self.epochs):
            activations = [X]
            for layer_idx in range(len(self.weights)):
                z = activations[-1] @ self.weights[layer_idx] + self.biases[layer_idx]
                if layer_idx == len(self.weights) - 1:
                    a = z
                else:
                    a = self._activation(z)
                activations.append(a)

            delta = 2.0 * (activations[-1] - y_scaled) / n_samples
            deltas = [delta]
            for layer_idx in range(len(self.weights) - 2, -1, -1):
                delta = (deltas[0] @ self.weights[layer_idx + 1].T)
                delta *= self._activation_derivative(activations[layer_idx + 1])
                deltas.insert(0, delta)

            for layer_idx in range(len(self.weights)):
                grad_w = activations[layer_idx].T @ deltas[layer_idx]
                grad_b = np.sum(deltas[layer_idx], axis=0, keepdims=True)
                grad_w = np.clip(grad_w, -5.0, 5.0)
                grad_b = np.clip(grad_b, -5.0, 5.0)
                self.weights[layer_idx] -= self.learning_rate * grad_w
                self.biases[layer_idx] -= self.learning_rate * grad_b
        return self

    def predict(self, X):
        a = X
        for layer_idx in range(len(self.weights)):
            z = a @ self.weights[layer_idx] + self.biases[layer_idx]
            a = z if layer_idx == len(self.weights) - 1 else self._activation(z)
        return (a.flatten() * self.y_std) + self.y_mean

# =============================================================
#  LOOP DE AVALIAÃ‡ÃƒO
# =============================================================

def evaluate_classifier(model, X, y_enc, n_classes, splits):
    """
    Roda K-Fold para um classificador e retorna mÃ©tricas mÃ©dias.
    """
    accs, precs, recs, specs, f1s = [], [], [], [], []
    t_train_list, t_test_list = [], []
    confusion_total = np.zeros((n_classes, n_classes), dtype=int)

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        X_tr_n, X_te_n = normalize(X_tr, X_te)

        t0 = time.perf_counter()
        model.fit(X_tr_n, y_tr)
        t_train_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = model.predict(X_te_n)
        t_test_list.append(time.perf_counter() - t0)

        accs.append(accuracy(y_te, y_pred))
        precs.append(precision_macro(y_te, y_pred, n_classes))
        recs.append(recall_macro(y_te, y_pred, n_classes))
        specs.append(specificity_macro(y_te, y_pred, n_classes))
        f1s.append(f1_macro(y_te, y_pred, n_classes))
        confusion_total += confusion_matrix(y_te, y_pred, n_classes)

    return {
        'acc':     (np.mean(accs),  np.std(accs)),
        'prec':    (np.mean(precs), np.std(precs)),
        'rec':     (np.mean(recs),  np.std(recs)),
        'spec':    (np.mean(specs), np.std(specs)),
        'f1':      (np.mean(f1s),   np.std(f1s)),
        't_train': (np.mean(t_train_list), np.std(t_train_list)),
        't_test':  (np.mean(t_test_list),  np.std(t_test_list)),
        'confusion': confusion_total,
    }


def evaluate_regressor(model, X, y, n_features, splits):
    """
    Roda K-Fold para um regressor e retorna mÃ©tricas mÃ©dias.
    """
    mses, rmses, maes, r2s, r2adjs = [], [], [], [], []
    t_train_list, t_test_list = [], []

    for train_idx, test_idx in splits:
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr_n, X_te_n = normalize(X_tr, X_te)

        t0 = time.perf_counter()
        model.fit(X_tr_n, y_tr)
        t_train_list.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        y_pred = model.predict(X_te_n)
        t_test_list.append(time.perf_counter() - t0)

        mses.append(mse(y_te, y_pred))
        rmses.append(rmse(y_te, y_pred))
        maes.append(mae(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
        r2adjs.append(r2_adjusted(y_te, y_pred, n_features))

    return {
        'mse':     (np.mean(mses),   np.std(mses)),
        'rmse':    (np.mean(rmses),  np.std(rmses)),
        'mae':     (np.mean(maes),   np.std(maes)),
        'r2':      (np.mean(r2s),    np.std(r2s)),
        'r2adj':   (np.mean(r2adjs), np.std(r2adjs)),
        't_train': (np.mean(t_train_list), np.std(t_train_list)),
        't_test':  (np.mean(t_test_list),  np.std(t_test_list)),
    }


# =============================================================
#  IMPRESSÃƒO DE RESULTADOS
# =============================================================

def fmt(mean, std):
    if max(abs(mean), abs(std)) >= 10000:
        return f"{mean:.3e} +/- {std:.3e}"
    return f"{mean:.4f} +/- {std:.4f}"


def print_classification_table(results, class_labels):
    print("\n" + "=" * 135)
    print("TABELA COMPARATIVA - CLASSIFICACAO (5-Fold Cross Validation)")
    print("=" * 135)
    header = (
        f"{'Classificador':<34} {'Acuracia':>17} {'Precisao':>17} "
        f"{'Sensib./Recall':>17} {'Especific.':>17} {'F1-Score':>17} "
        f"{'T.Treino(s)':>14} {'T.Teste(s)':>13}"
    )
    print(header)
    print("-" * 135)
    for name, r in results.items():
        print(
            f"{name:<34} {fmt(*r['acc']):>17} {fmt(*r['prec']):>17} "
            f"{fmt(*r['rec']):>17} {fmt(*r['spec']):>17} {fmt(*r['f1']):>17} "
            f"{fmt(*r['t_train']):>14} {fmt(*r['t_test']):>13}"
        )
    print("=" * 135)
    print_confusion_matrices(results, class_labels)


def print_confusion_matrices(results, class_labels):
    labels = [str(label) for label in class_labels]
    label_width = max(8, max(len(label) for label in labels) + 2)

    print("\nMATRIZES DE CONFUSAO ACUMULADAS NOS FOLDS")
    print("Linhas = classe real | Colunas = classe prevista")
    for name, r in results.items():
        matrix = r['confusion']
        print(f"\n{name}")
        print(" " * label_width + "".join(f"{label:>{label_width}}" for label in labels))
        for label, row in zip(labels, matrix):
            print(f"{label:>{label_width}}" + "".join(f"{value:>{label_width}d}" for value in row))

    best_f1 = max(results, key=lambda name: results[name]['f1'][0])
    best_acc = max(results, key=lambda name: results[name]['acc'][0])
    fastest_test = min(results, key=lambda name: results[name]['t_test'][0])
    print("\nResumo para discussao:")
    print(f"  Melhor F1-score: {best_f1} ({results[best_f1]['f1'][0]:.4f})")
    print(f"  Melhor acuracia: {best_acc} ({results[best_acc]['acc'][0]:.4f})")
    print(f"  Menor tempo de teste: {fastest_test} ({results[fastest_test]['t_test'][0]:.4f}s)")


def print_regression_table(results):
    print("\n" + "=" * 150)
    print("TABELA COMPARATIVA - REGRESSAO (5-Fold Cross Validation)")
    print("=" * 150)
    header = (
        f"{'Regressor':<38} {'MSE':>17} {'RMSE':>17} {'MAE':>17} "
        f"{'R2':>17} {'R2 Ajustado':>17} {'T.Treino(s)':>14} {'T.Teste(s)':>13}"
    )
    print(header)
    print("-" * 150)
    for name, r in results.items():
        print(
            f"{name:<38} {fmt(*r['mse']):>17} {fmt(*r['rmse']):>17} "
            f"{fmt(*r['mae']):>17} {fmt(*r['r2']):>17} {fmt(*r['r2adj']):>17} "
            f"{fmt(*r['t_train']):>14} {fmt(*r['t_test']):>13}"
        )
    print("=" * 150)
    best_rmse = min(results, key=lambda name: results[name]['rmse'][0])
    best_r2 = max(results, key=lambda name: results[name]['r2'][0])
    fastest_train = min(results, key=lambda name: results[name]['t_train'][0])
    print("\nResumo para discussao:")
    print(f"  Menor RMSE: {best_rmse} ({results[best_rmse]['rmse'][0]:.4f})")
    print(f"  Melhor R2: {best_r2} ({results[best_r2]['r2'][0]:.4f})")
    print(f"  Menor tempo de treino: {fastest_train} ({results[fastest_train]['t_train'][0]:.4f}s)")


# =============================================================
#  MAIN â€” CLASSIFICAÃ‡ÃƒO
# =============================================================

def run_classification(arff_path, k_folds=5, k_neighbors=5):
    print(f"\n{'#'*60}")
    print(f"  CLASSIFICACAO - Dataset: {arff_path}")
    print(f"{'#'*60}")

    print("Carregando dataset...")
    X, y_raw, feature_names = parse_arff(arff_path)
    y_enc, classes = encode_labels(y_raw)
    n_classes = len(classes)
    n_features = X.shape[1]

    print(f"  Instancias : {X.shape[0]}")
    print(f"  Atributos  : {n_features}")
    print(f"  Classes    : {classes}")

    splits = kfold_indices(len(y_enc), k=k_folds)

    classifiers = {
        f"KNN k={k_neighbors} Euclidiana":    KNNClassifier(k=k_neighbors, metric='euclidean'),
        f"KNN k={k_neighbors} Manhattan":     KNNClassifier(k=k_neighbors, metric='manhattan'),
        "Naive Bayes Univariado":             NaiveBayesUni(),
        "Naive Bayes Multivariado":           NaiveBayesMulti(),
        "Perceptron lr=0.01 ep=50":           PerceptronSimples(learning_rate=0.01, epochs=50, seed=11),
        "Perceptron lr=0.01 ep=100":          PerceptronSimples(learning_rate=0.01, epochs=100, seed=11),
        "Perceptron lr=0.05 ep=100":          PerceptronSimples(learning_rate=0.05, epochs=100, seed=11),
        "Perceptron lr=0.10 ep=200":          PerceptronSimples(learning_rate=0.10, epochs=200, seed=11),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nAvaliando: {name} ...")
        results[name] = evaluate_classifier(clf, X, y_enc, n_classes, splits)
        print(f"  Acuracia media: {results[name]['acc'][0]:.4f}")

    print_classification_table(results, classes)
    return results


# =============================================================
#  MAIN â€” REGRESSÃƒO
# =============================================================

def run_regression(data_path, is_arff=False, target_name=None, drop_names=None,
                   k_folds=5, k_neighbors=5):
    print(f"\n{'#'*60}")
    print(f"  REGRESSAO - Dataset: {data_path}")
    print(f"{'#'*60}")

    print("Carregando dataset...")
    if is_arff:
        X, y_raw, feature_names = parse_arff(
            data_path,
            target_name=target_name,
            drop_names=drop_names
        )
        # Para regressao em ARFF, converte rotulo para float
        try:
            y = y_raw.astype(np.float64)
        except ValueError:
            # Se nao for numerico, codifica como inteiro
            y_enc, _ = encode_labels(y_raw)
            y = y_enc.astype(np.float64)
    else:
        X, y, feature_names = parse_csv(data_path)

    n_features = X.shape[1]
    print(f"  Instancias : {X.shape[0]}")
    print(f"  Atributos  : {n_features}")
    print(f"  Alvo (y)   : min={y.min():.3f}, max={y.max():.3f}, media={y.mean():.3f}")

    splits = kfold_indices(len(y), k=k_folds)

    regressors = {
        f"KNN k={k_neighbors} Euclidiana": KNNRegressor(k=k_neighbors, metric='euclidean'),
        f"KNN k={k_neighbors} Manhattan":  KNNRegressor(k=k_neighbors, metric='manhattan'),
        "Regressao Linear Multipla":        LinearRegression(),
        "MLP (8) relu lr=0.010 ep=250":     MLPRegressor(hidden_layers=(8,), learning_rate=0.010, epochs=250, activation='relu', seed=21),
        "MLP (16) tanh lr=0.010 ep=300":    MLPRegressor(hidden_layers=(16,), learning_rate=0.010, epochs=300, activation='tanh', seed=22),
        "MLP (16,8) relu lr=0.005 ep=350":  MLPRegressor(hidden_layers=(16, 8), learning_rate=0.005, epochs=350, activation='relu', seed=23),
        "MLP (32,16) relu lr=0.003 ep=400": MLPRegressor(hidden_layers=(32, 16), learning_rate=0.003, epochs=400, activation='relu', seed=24),
    }

    results = {}
    for name, reg in regressors.items():
        print(f"\nAvaliando: {name} ...")
        results[name] = evaluate_regressor(reg, X, y, n_features, splits)
        print(f"  R2 medio: {results[name]['r2'][0]:.4f}")

    print_regression_table(results)
    return results


# =============================================================
#  PONTO DE ENTRADA
# =============================================================

if __name__ == "__main__":
    import os

    # -------------------------------------------------------
    # AJUSTE AQUI OS CAMINHOS DOS SEUS DATASETS
    # -------------------------------------------------------
    CLASSIFICATION_ARFF = "phpV5QYya.arff"   # dataset de classificacao
    REGRESSION_DATA     = "dataset.arff"     # dataset de regressao (CSV ou ARFF)
    REGRESSION_IS_ARFF  = True               # True se o dataset de regressao for ARFF
    REGRESSION_TARGET   = "price"
    REGRESSION_DROP     = ["id"]
    K_FOLDS      = 5
    K_NEIGHBORS  = 5
    # -------------------------------------------------------

    # Classificacao
    if os.path.exists(CLASSIFICATION_ARFF):
        run_classification(CLASSIFICATION_ARFF, k_folds=K_FOLDS, k_neighbors=K_NEIGHBORS)
    else:
        print(f"[AVISO] Arquivo '{CLASSIFICATION_ARFF}' nao encontrado.")

    # Regressao
    if os.path.exists(REGRESSION_DATA):
        run_regression(REGRESSION_DATA, is_arff=REGRESSION_IS_ARFF,
                       target_name=REGRESSION_TARGET, drop_names=REGRESSION_DROP,
                       k_folds=K_FOLDS, k_neighbors=K_NEIGHBORS)
    else:
        print(f"[AVISO] Arquivo '{REGRESSION_DATA}' nao encontrado.")
