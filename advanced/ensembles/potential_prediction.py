from sklearn.base import BaseEstimator, TransformerMixin
from scipy.ndimage import shift, center_of_mass
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from scipy.ndimage import center_of_mass, shift
import warnings
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings('ignore')


class PotentialTransformer(BaseEstimator, TransformerMixin):
    """
    Улучшенный трансформер для потенциалов:
    - центрирование потенциала (перенос центра масс в центр изображения)
    - опциональная обрезка фиксированной области вокруг центра (убирает дальний фон)
    - нормализация значений в [0, 1]
    """

    def __init__(self, target_size=(256, 256), normalize=True, center_potential=True,
                 crop_size=220, background_value=20):
        """
        :param target_size: целевой размер изображения (height, width)
        :param normalize: нормализовать значения в [0, 1]
        :param center_potential: центрировать потенциал
        :param crop_size: если не None, вырезать квадрат crop_size x crop_size вокруг центра
                          (должен быть ≤ target_size[0])
        :param background_value: значение фона (по умолчанию 20)
        """
        self.target_size = target_size
        self.normalize = normalize
        self.center_potential = center_potential
        self.crop_size = crop_size
        self.background_value = background_value
        self.original_shape_ = None
        self.transformed_shape_ = None

    def _find_potential_center(self, potential):
        """Находит центр потенциала (центр масс области с низкими значениями)"""
        # Порог – нижние 10% значений
        threshold = np.percentile(potential, 10)
        mask = potential < threshold
        if np.sum(mask) > 0:
            center = center_of_mass(mask)
        else:
            # Если маска пуста, используем глобальный минимум
            min_pos = np.unravel_index(np.argmin(potential), potential.shape)
            center = (float(min_pos[0]), float(min_pos[1]))
        return center

    def _center_potential(self, potential):
        """Переносит центр потенциала в центр изображения"""
        cy, cx = self._find_potential_center(potential)
        target_cy, target_cx = self.target_size[0] / 2, self.target_size[1] / 2
        shift_y = target_cy - cy
        shift_x = target_cx - cx
        centered = shift(potential, (shift_y, shift_x),
                         mode='constant', cval=self.background_value)
        return centered

    def _crop_and_paste(self, potential):
        """
        Вырезает квадрат crop_size x crop_size вокруг центра изображения
        и вставляет его в центр нового изображения target_size.
        Области за пределами вырезки заполняются background_value.
        """
        if self.crop_size is None or self.crop_size >= self.target_size[0]:
            return potential  # нет обрезки или она больше размера

        h, w = self.target_size
        c = self.crop_size
        # Центр изображения
        center_y, center_x = h // 2, w // 2

        # Координаты вырезаемой области
        y_start = center_y - c // 2
        y_end = y_start + c
        x_start = center_x - c // 2
        x_end = x_start + c

        # Вырезаем (если выходит за границы – обрезаем, но такого не должно быть,
        # так как потенциал центрирован и crop_size ≤ target_size)
        crop = potential[y_start:y_end, x_start:x_end]

        # Создаём новое изображение, заполненное фоном
        new_img = np.full(self.target_size,
                          self.background_value, dtype=potential.dtype)

        # Вставляем вырезку в центр
        new_y_start = center_y - crop.shape[0] // 2
        new_x_start = center_x - crop.shape[1] // 2
        new_img[new_y_start:new_y_start + crop.shape[0],
                new_x_start:new_x_start + crop.shape[1]] = crop

        return new_img

    def _normalize(self, potential):
        """Нормализует значения в [0, 1]"""
        if not self.normalize:
            return potential
        min_val = potential.min()
        max_val = potential.max()
        if max_val > min_val:
            return (potential - min_val) / (max_val - min_val)
        return potential

    def fit(self, X, y=None):
        """Запоминает исходную форму"""
        self.original_shape_ = X.shape[1:] if X.ndim == 3 else None
        return self

    def transform(self, X):
        """
        Преобразует массив изображений X (N, H, W) в массив признаков (N, H*W)
        с применением центрирования, обрезки и нормализации.
        """
        transformed = []
        for img in X:
            proc = img.copy().astype(float)
            if self.center_potential:
                proc = self._center_potential(proc)
            if self.crop_size is not None:
                proc = self._crop_and_paste(proc)
            transformed.append(proc)

        transformed = np.array(transformed)
        self.transformed_shape_ = transformed.shape[1:]
        return transformed.reshape(transformed.shape[0], -1)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X_flat):
        """Обратное преобразование из плоского вида в изображения"""
        if self.transformed_shape_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return X_flat.reshape(-1, *self.transformed_shape_)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('et', ExtraTreesRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        max_features=0.4,
    ))])

    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
