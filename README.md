# Potato Disease Classification with CNNs and Transfer Learning

This project implements **deep learning models for classifying potato leaf images** into three categories: *Potato Early Blight*, *Potato Late Blight*, and *Potato Healthy*. It demonstrates the use of custom Convolutional Neural Networks (CNNs) as well as **transfer learning** with EfficientNetB0 using TensorFlow and Keras.

## Features

- **Image loading and preprocessing** with TensorFlow Datasets
- **Data augmentation** for improved generalization
- **Two custom CNN architectures** with increasing complexity
- **Transfer learning** with EfficientNetB0 (pre-trained on ImageNet)
- **Training/validation/test split** and performance visualization
- **Early stopping** to prevent overfitting

## Dataset

- **Source:** `PlantVillage` potato leaf dataset
- **Classes:** 
  - Potato___Early_blight
  - Potato___Late_blight
  - Potato___healthy
- **Total Images:** 2,152
- **Image Size:** 256x256 (resized as needed)

## Requirements

- Python 3.x
- TensorFlow 2.x
- Matplotlib

Install dependencies with:

`pip install tensorflow matplotlib`

## Usage

### 1. Data Preparation

- The script checks for the dataset at `/kaggle/input/potato-disease-blight/PlantVillage` or `potato-blight-disease/PlantVillage`.
- Images are loaded into a TensorFlow Dataset, shuffled, and split into train, validation, and test sets.

### 2. Preprocessing & Augmentation

- **Scaling:** Pixel values are normalized to .
- **Resizing:** All images are resized to 256x256.
- **Augmentation:** Random flips and rotations are applied to training data only.

### 3. Model Architectures

| Model                | Description                               | Params    | Training Acc. | Test Acc. | Notes           |
|----------------------|-------------------------------------------|-----------|---------------|-----------|-----------------|
| **Simple CNN**       | 6 Conv2D layers, Dense, Softmax           | 551K      | 99.52%        | 98.82%    | Good fit        |
| **Deeper CNN**       | More filters, BatchNorm, Dropout          | 1.27M     | 98.73%        | 72.48%    | Overfitting     |
| **EfficientNetB0**   | Transfer learning, frozen base, Dropout   | 4.3M      | 98.30%        | 98.43%    | Fast, robust    |

### 4. Training & Evaluation

- Models are compiled with Adam optimizer and sparse categorical crossentropy loss.
- Training is performed with validation monitoring.
- Early stopping is used for EfficientNetB0.
- Accuracy and loss curves are plotted for each model.

### 5. Results

- **Simple CNN**: High accuracy, well-balanced, no overfitting.
- **Deeper CNN**: Overfits due to excessive parameters.
- **EfficientNetB0**: Achieves high accuracy quickly and robustly with transfer learning.

## Example: Training a Simple CNN

```python
model = models.Sequential([
resize_and_rescale,
layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
# ... (other layers) ...
layers.Dense(3, activation='softmax'),
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, validation_data=val_ds, epochs=50)
```

## Visualization

- **Training and validation accuracy/loss** are plotted after each run for model assessment.
- Example code for plotting:

```python
plt.plot(range(EPOCHS), history.history['accuracy'], label='Training Accuracy')
plt.plot(range(EPOCHS), history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
```

## Key Takeaways

- **Data augmentation** and **proper model complexity** are crucial for generalization.
- **Transfer learning** (EfficientNetB0) provides state-of-the-art results with less overfitting and faster convergence.
- **Early stopping** is effective for preventing overfitting in deep models.

## References

- Based on TensorFlow and Keras best practices for image classification.

## Author

- Adapted for educational and research purposes.

: Source code and details extracted from the provided file.
