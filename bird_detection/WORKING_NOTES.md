
# ADD TO TESTS:

* The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each image, and should be in 0-1 range. Different images can have different sizes.

The behavior of the model changes depending on if it is in training or evaluation mode.

During training, the model expects both the input tensors and a targets (list of dictionary), containing:

boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

labels (Int64Tensor[N]): the class label for each ground-truth box

The model returns a Dict[Tensor] during training, containing the classification and regression losses for both the RPN and the R-CNN.

During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as follows, where N is the number of detections:

boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.

labels (Int64Tensor[N]): the predicted labels for each detection

scores (Tensor[N]): the scores of each detection


* Check right bb denormalization, that нет превышений размера картинки (можно визуально оценить смещение)

## Training CLI reminders

* Запуск с Hydra: `python -m bird_detection.detection.train training.epochs=1 model=ssd augmentations.train.advanced.mosaic.enabled=true`.
* TensorBoard логи живут в `${hydra:run.dir}/tensorboard`, output-модель — `${hydra:run.dir}/trained_model`.

## Testing reminders

* Pytests для всего проекта: `pytest`. Полезные подсuites:
  * `pytest tests/test_augmentations.py` — проверяет `DetectionTransformPipeline` (0-1 тензоры, нормализация, ресайз, интеграция с Albumentations, поведение при удалении боксов).
  * `pytest tests/test_data_scripts.py` — валидирует экспорт белок и CUB: структура файлов, форматы `labels_pixel`, ресайз, обрезка, CSV с отображением классов.
  * `pytest tests/test_dataloaders.py` — sanity-check датасет/колайт и `DatasetMetadata` (ограничение диапазонов, область боксов, уникальные ID птиц/белок).
  * `pytest tests/test_predictions.py` — форматирование выводов модели в API (`bbox`, фильтрация по порогу, fallback `not_found`).
* Визуальная проверка денормализации: `python -m bird_detection.data_scripts.visualize_bboxes --images-dir bird_detection/data/selected/birds/<class_dir>/images --labels-dir bird_detection/data/selected/birds/<class_dir>/labels_pixel --output-dir outputs/bbox_viz/<class_dir> --limit 5 --shuffle`.



### Добавление новых данных или моделей в DVC

```bash
# Добавить директорию в DVC
dvc add path/to/data

# Зафиксировать в git
git add path/to/data.dvc .gitignore
git commit -m "Add new dataset version"

# Отправить в remote
dvc push
```