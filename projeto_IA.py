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
    """Calcula TP, FP, FN para cada classe."""
    TP = np.zeros(n_classes)
    FP = np.zeros(n_classes)
    FN = np.zeros(n_classes)
    for c in range(n_classes):
        TP[c] = np.sum((y_pred == c) & (y_true == c))
        FP[c] = np.sum((y_pred == c) & (y_true != c))
        FN[c] = np.sum((y_pred != c) & (y_true == c))
    return TP, FP, FN


def precision_macro(y_true, y_pred, n_classes):
    TP, FP, _ = confusion_per_class(y_true, y_pred, n_classes)
    denom = TP + FP
    per_class = np.where(denom > 0, TP / denom, 0.0)
    return per_class.mean()


def recall_macro(y_true, y_pred, n_classes):
    TP, _, FN = confusion_per_class(y_true, y_pred, n_classes)
    denom = TP + FN
    per_class = np.where(denom > 0, TP / denom, 0.0)
    return per_class.mean()


def f1_macro(y_true, y_pred, n_classes):
    p = precision_macro(y_true, y_pred, n_classes)
    r = recall_macro(y_true, y_pred, n_classes)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# =============================================================
#  MÃ‰TRICAS DE REGRESSÃƒO  (sem sklearn)
# =============================================================

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


# =============================================================
#  LOOP DE AVALIAÃ‡ÃƒO
# =============================================================

def evaluate_classifier(model, X, y_enc, n_classes, splits):
    """
    Roda K-Fold para um classificador e retorna mÃ©tricas mÃ©dias.
    """
    accs, precs, recs, f1s = [], [], [], []
    t_train_list, t_test_list = [], []

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
        f1s.append(f1_macro(y_te, y_pred, n_classes))

    return {
        'acc':     (np.mean(accs),  np.std(accs)),
        'prec':    (np.mean(precs), np.std(precs)),
        'rec':     (np.mean(recs),  np.std(recs)),
        'f1':      (np.mean(f1s),   np.std(f1s)),
        't_train': (np.mean(t_train_list), np.std(t_train_list)),
        't_test':  (np.mean(t_test_list),  np.std(t_test_list)),
    }


def evaluate_regressor(model, X, y, n_features, splits):
    """
    Roda K-Fold para um regressor e retorna mÃ©tricas mÃ©dias.
    """
    r2s, r2adjs = [], []
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

        r2s.append(r2_score(y_te, y_pred))
        r2adjs.append(r2_adjusted(y_te, y_pred, n_features))

    return {
        'r2':      (np.mean(r2s),    np.std(r2s)),
        'r2adj':   (np.mean(r2adjs), np.std(r2adjs)),
        't_train': (np.mean(t_train_list), np.std(t_train_list)),
        't_test':  (np.mean(t_test_list),  np.std(t_test_list)),
    }


# =============================================================
#  IMPRESSÃƒO DE RESULTADOS
# =============================================================

def fmt(mean, std):
    return f"{mean:.4f} Â± {std:.4f}"


def print_classification_table(results):
    print("\n" + "=" * 100)
    print("TABELA COMPARATIVA â€” CLASSIFICAÃ‡ÃƒO (5-Fold Cross Validation)")
    print("=" * 100)
    header = f"{'Classificador':<35} {'AcurÃ¡cia':>18} {'PrecisÃ£o':>18} {'Recall':>18} {'F1-Score':>18} {'T.Treino(s)':>14} {'T.Teste(s)':>13}"
    print(header)
    print("-" * 100)
    for name, r in results.items():
        print(f"{name:<35} {fmt(*r['acc']):>18} {fmt(*r['prec']):>18} {fmt(*r['rec']):>18} {fmt(*r['f1']):>18} {fmt(*r['t_train']):>14} {fmt(*r['t_test']):>13}")
    print("=" * 100)


def print_regression_table(results):
    print("\n" + "=" * 75)
    print("TABELA COMPARATIVA â€” REGRESSÃƒO (5-Fold Cross Validation)")
    print("=" * 75)
    header = f"{'Regressor':<35} {'RÂ²':>16} {'RÂ² Ajustado':>18} {'T.Treino(s)':>14} {'T.Teste(s)':>13}"
    print(header)
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<35} {fmt(*r['r2']):>16} {fmt(*r['r2adj']):>18} {fmt(*r['t_train']):>14} {fmt(*r['t_test']):>13}")
    print("=" * 75)


# =============================================================
#  MAIN â€” CLASSIFICAÃ‡ÃƒO
# =============================================================

def run_classification(arff_path, k_folds=5, k_neighbors=5):
    print(f"\n{'#'*60}")
    print(f"  CLASSIFICAÃ‡ÃƒO â€” Dataset: {arff_path}")
    print(f"{'#'*60}")

    print("Carregando dataset...")
    X, y_raw, feature_names = parse_arff(arff_path)
    y_enc, classes = encode_labels(y_raw)
    n_classes = len(classes)
    n_features = X.shape[1]

    print(f"  InstÃ¢ncias : {X.shape[0]}")
    print(f"  Atributos  : {n_features}")
    print(f"  Classes    : {classes}")

    splits = kfold_indices(len(y_enc), k=k_folds)

    classifiers = {
        f"KNN k={k_neighbors} Euclidiana":    KNNClassifier(k=k_neighbors, metric='euclidean'),
        f"KNN k={k_neighbors} Manhattan":     KNNClassifier(k=k_neighbors, metric='manhattan'),
        "Naive Bayes Univariado":             NaiveBayesUni(),
        "Naive Bayes Multivariado":           NaiveBayesMulti(),
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"\nAvaliando: {name} ...")
        results[name] = evaluate_classifier(clf, X, y_enc, n_classes, splits)
        print(f"  AcurÃ¡cia mÃ©dia: {results[name]['acc'][0]:.4f}")

    print_classification_table(results)
    return results


# =============================================================
#  MAIN â€” REGRESSÃƒO
# =============================================================

def run_regression(data_path, is_arff=False, target_name=None, drop_names=None,
                   k_folds=5, k_neighbors=5):
    print(f"\n{'#'*60}")
    print(f"  REGRESSÃƒO â€” Dataset: {data_path}")
    print(f"{'#'*60}")

    print("Carregando dataset...")
    if is_arff:
        X, y_raw, feature_names = parse_arff(
            data_path,
            target_name=target_name,
            drop_names=drop_names
        )
        # Para regressÃ£o em ARFF, converte rÃ³tulo para float
        try:
            y = y_raw.astype(np.float64)
        except ValueError:
            # Se nÃ£o for numÃ©rico, codifica como inteiro
            y_enc, _ = encode_labels(y_raw)
            y = y_enc.astype(np.float64)
    else:
        X, y, feature_names = parse_csv(data_path)

    n_features = X.shape[1]
    print(f"  InstÃ¢ncias : {X.shape[0]}")
    print(f"  Atributos  : {n_features}")
    print(f"  Alvo (y)   : min={y.min():.3f}, max={y.max():.3f}, mÃ©dia={y.mean():.3f}")

    splits = kfold_indices(len(y), k=k_folds)

    regressors = {
        f"KNN k={k_neighbors} Euclidiana": KNNRegressor(k=k_neighbors, metric='euclidean'),
        f"KNN k={k_neighbors} Manhattan":  KNNRegressor(k=k_neighbors, metric='manhattan'),
        "RegressÃ£o Linear MÃºltipla":       LinearRegression(),
    }

    results = {}
    for name, reg in regressors.items():
        print(f"\nAvaliando: {name} ...")
        results[name] = evaluate_regressor(reg, X, y, n_features, splits)
        print(f"  RÂ² mÃ©dio: {results[name]['r2'][0]:.4f}")

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
    CLASSIFICATION_ARFF = "phpV5QYya.arff"   # dataset de classificaÃ§Ã£o
    REGRESSION_DATA     = "dataset.arff"          # dataset de regressÃ£o (CSV ou ARFF)
    REGRESSION_IS_ARFF  = True               # True se o dataset de regressÃ£o for ARFF
    REGRESSION_TARGET   = "price"
    REGRESSION_DROP     = ["id"]
    K_FOLDS      = 5
    K_NEIGHBORS  = 5
    # -------------------------------------------------------

    # ClassificaÃ§Ã£o
    if os.path.exists(CLASSIFICATION_ARFF):
        run_classification(CLASSIFICATION_ARFF, k_folds=K_FOLDS, k_neighbors=K_NEIGHBORS)
    else:
        print(f"[AVISO] Arquivo '{CLASSIFICATION_ARFF}' nÃ£o encontrado.")

    # RegressÃ£o
    if os.path.exists(REGRESSION_DATA):
        run_regression(REGRESSION_DATA, is_arff=REGRESSION_IS_ARFF,
                       target_name=REGRESSION_TARGET, drop_names=REGRESSION_DROP,
                       k_folds=K_FOLDS, k_neighbors=K_NEIGHBORS)
    else:
        print(f"[AVISO] Arquivo '{REGRESSION_DATA}' nÃ£o encontrado.")
