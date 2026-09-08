import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                   BatchNormalization, Flatten, Input, Conv2D, 
                                   MaxPooling2D)
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                       ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from pathlib import Path
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# CONFIGURATION
# =====================================================================
class Config:
    # Image parameters
    IMG_HEIGHT = 224  # VGG16 requires 224x224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS_INITIAL = 10
    EPOCHS_FINE_TUNE = 10
    LEARNING_RATE = 1e-4
    FINE_TUNE_LR = 1e-5
    
    # Dataset split ratios
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Model parameters
    DROPOUT_RATE = 0.5
    DENSE_UNITS = 512
    L2_REGULARIZATION = 0.001
    
    # Training parameters
    EARLY_STOPPING_PATIENCE = 5
    REDUCE_LR_PATIENCE = 3
    REDUCE_LR_FACTOR = 0.2
    
    # Custom augmentation parameters
    ROTATION_RANGE = 0.2
    ZOOM_RANGE = 0.2
    BRIGHTNESS_RANGE = [0.8, 1.2]
    CONTRAST_RANGE = [0.8, 1.2]
    
    # Paths
    BASE_DIR = 'abc'
    MODEL_SAVE_PATH = 'models'
    LOG_DIR = 'logs'
    RESULTS_DIR = 'results'

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def setup_directories():
    """Create necessary directories for saving models and results"""
    dirs = [Config.MODEL_SAVE_PATH, Config.LOG_DIR, Config.RESULTS_DIR]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dirs

def get_class_info(directory):
    """Get class names from the dataset directory"""
    dataset = image_dataset_from_directory(
        directory,
        image_size=(Config.IMG_HEIGHT, Config.IMG_WIDTH),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        seed=123
    )
    class_names = dataset.class_names
    num_classes = len(class_names)
    print(f'✓ Found {num_classes} classes: {class_names}')
    return class_names, num_classes

# =====================================================================
# DATA AUGMENTATION
# =====================================================================
def custom_augment(image, label):
    """Custom data augmentation with more transformations"""
    # Brightness and contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    
    # Flips
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random rotation (simulated with wrap)
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)
    
    # Random saturation and hue
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)
    
    # Ensure proper size
    image = tf.image.resize(image, [Config.IMG_HEIGHT, Config.IMG_WIDTH])
    
    return image, label

def one_hot_encode(image, label):
    """Convert labels to one-hot encoding"""
    label = to_categorical(label, num_classes=num_classes)
    return image, label

def normalize_image(image, label):
    """Normalize pixel values to [0, 1]"""
    return image / 255.0, label

# =====================================================================
# DATA LOADING
# =====================================================================
def load_dataset(directory, batch_size, subset=None, validation_split=None):
    """Load dataset with all preprocessing steps"""
    kwargs = {'directory': directory,
              'image_size': (Config.IMG_HEIGHT, Config.IMG_WIDTH),
              'batch_size': batch_size,
              'shuffle': True,
              'seed': 123}
    
    if subset:
        kwargs['subset'] = subset
        kwargs['validation_split'] = validation_split
    
    dataset = image_dataset_from_directory(**kwargs)
    
    # Apply preprocessing
    dataset = dataset.map(custom_augment)
    dataset = dataset.map(one_hot_encode)
    dataset = dataset.map(normalize_image)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# =====================================================================
# MODEL CREATION
# =====================================================================
def create_model(model_type='vgg16', num_classes=2):
    """Create model with different base architectures"""
    base_models = {
        'vgg16': VGG16,
        'resnet50': ResNet50,
        'mobilenet': MobileNetV2
    }
    
    input_shape = (Config.IMG_HEIGHT, Config.IMG_WIDTH, 3)
    
    if model_type == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=input_shape)
    elif model_type == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=input_shape)
    elif model_type == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Freeze base model initially
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(Config.DENSE_UNITS, activation='relu', 
              kernel_regularizer=l2(Config.L2_REGULARIZATION)),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        Dense(Config.DENSE_UNITS // 2, activation='relu', 
              kernel_regularizer=l2(Config.L2_REGULARIZATION)),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE * 0.8),
        Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model

# =====================================================================
# CALLBACKS
# =====================================================================
def get_callbacks(model_name):
    """Create training callbacks"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', 
                     patience=Config.EARLY_STOPPING_PATIENCE, 
                     restore_best_weights=True,
                     verbose=1),
        
        ReduceLROnPlateau(monitor='val_loss', 
                         factor=Config.REDUCE_LR_FACTOR, 
                         patience=Config.REDUCE_LR_PATIENCE, 
                         min_lr=1e-7,
                         verbose=1),
        
        ModelCheckpoint(f'{Config.MODEL_SAVE_PATH}/{model_name}_{timestamp}_best.keras',
                       monitor='val_accuracy',
                       save_best_only=True,
                       verbose=1),
        
        TensorBoard(log_dir=f'{Config.LOG_DIR}/{model_name}_{timestamp}',
                   histogram_freq=1)
    ]
    
    return callbacks

# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================
def plot_training_history(history, title='Training History', save=True):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', 
                 color='#2ecc71', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', 
                 color='#e74c3c', linewidth=2)
    axes[0].set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', 
                 color='#2ecc71', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', 
                 color='#e74c3c', linewidth=2)
    axes[1].set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{Config.RESULTS_DIR}/training_history_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        print(f'✓ Training history saved to {Config.RESULTS_DIR}/')
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{Config.RESULTS_DIR}/confusion_matrix_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        print(f'✓ Confusion matrix saved to {Config.RESULTS_DIR}/')
    
    plt.show()

def plot_precision_recall_curve(y_true, y_scores, class_names, save=True):
    """Plot precision-recall curves for multi-class"""
    plt.figure(figsize=(10, 8))
    
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], 
                                                       y_scores[:, i])
        plt.plot(recall, precision, lw=2, 
                label=f'{class_names[i]} (AP={auc(recall, precision):.2f})')
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'{Config.RESULTS_DIR}/precision_recall_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        print(f'✓ PR curve saved to {Config.RESULTS_DIR}/')
    
    plt.show()

# =====================================================================
# EVALUATION FUNCTIONS
# =====================================================================
def evaluate_model(model, dataset, class_names, dataset_name='Test'):
    """Comprehensive model evaluation"""
    print(f'\n{"="*50}')
    print(f'EVALUATING ON {dataset_name.upper()} DATASET')
    print(f'{"="*50}')
    
    # Get predictions
    y_true = []
    y_pred = []
    y_scores = []
    
    for batch_images, batch_labels in dataset:
        predictions = model.predict(batch_images, verbose=0)
        true_labels = np.argmax(batch_labels.numpy(), axis=1)
        pred_labels = np.argmax(predictions, axis=1)
        
        y_true.extend(true_labels)
        y_pred.extend(pred_labels)
        y_scores.extend(predictions)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    # Classification report
    print('\n📊 Classification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Precision-Recall curves
    if len(class_names) == 2:
        # For binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
        print(f'\n📈 Average Precision: {auc(recall, precision):.4f}')
    else:
        # For multi-class
        plot_precision_recall_curve(y_true, y_scores, class_names)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(label_binarize(y_true, classes=range(len(class_names))), 
                                     y_scores, average='weighted')
        print(f'\n📈 Weighted Average Precision: {ap:.4f}')
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f'\n✅ {dataset_name} Accuracy: {accuracy * 100:.2f}%')
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
    }

# =====================================================================
# PREDICTION FUNCTION
# =====================================================================
def predict_single_image(model, image_path, class_names):
    """Predict class for a single image"""
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.predict(img, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class] * 100
        
        print(f'\n🔍 Image: {os.path.basename(image_path)}')
        print(f'📋 Predicted class: {class_names[predicted_class]}')
        print(f'🎯 Confidence: {confidence:.2f}%')
        print('\n📊 Class probabilities:')
        for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
            print(f'   {class_name}: {prob * 100:.2f}%')
        
        return predicted_class, confidence, predictions
    
    except Exception as e:
        print(f'❌ Error processing image: {e}')
        return None, None, None

def predict_batch(model, image_dir, class_names):
    """Predict for multiple images in a directory"""
    print(f'\n🔍 Analyzing images in {image_dir}...')
    
    results = []
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        predicted_class, confidence, _ = predict_single_image(model, img_path, class_names)
        if predicted_class is not None:
            results.append({
                'image': img_file,
                'predicted_class': class_names[predicted_class],
                'confidence': confidence
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'{Config.RESULTS_DIR}/predictions_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n✅ Results saved to {csv_path}')
    
    return df

# =====================================================================
# MAIN TRAINING PIPELINE
# =====================================================================
def main():
    """Main training pipeline"""
    print('='*60)
    print(' LUNG CANCER DETECTION - TRAINING PIPELINE')
    print('='*60)
    
    # Setup directories
    setup_directories()
    
    # Global variables
    global class_names, num_classes
    
    # Get class info
    print('\n📂 Loading dataset information...')
    class_names, num_classes = get_class_info(Config.BASE_DIR)
    
    # Load datasets
    print('\n📥 Loading datasets...')
    train_dataset = load_dataset(Config.BASE_DIR, Config.BATCH_SIZE, 
                                subset='training', validation_split=0.3)
    validation_dataset = load_dataset(Config.BASE_DIR, Config.BATCH_SIZE, 
                                     subset='validation', validation_split=0.5)
    test_dataset = load_dataset(Config.BASE_DIR, Config.BATCH_SIZE, 
                               subset='validation', validation_split=0.5)
    
    # Create model
    print('\n🏗️ Creating VGG16 model...')
    model, base_model = create_model('vgg16', num_classes)
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE),
                 loss='categorical_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()])
    
    print('\n📋 Model Summary:')
    model.summary()
    
    # Plot model architecture
    plot_model(model, to_file=f'{Config.RESULTS_DIR}/model_architecture.png', 
              show_shapes=True, show_layer_names=True)
    print(f'✓ Model architecture saved to {Config.RESULTS_DIR}/')
    
    # Train initial model
    print('\n🚀 Starting initial training...')
    callbacks = get_callbacks('vgg16')
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=Config.EPOCHS_INITIAL,
        callbacks=callbacks
    )
    
    # Plot initial training history
    plot_training_history(history, 'Initial Training')
    
    # Fine-tuning
    print('\n🔧 Starting fine-tuning...')
    base_model.trainable = True
    
    # Freeze first 15 layers
    for layer in base_model.layers[:15]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(optimizer=Adam(learning_rate=Config.FINE_TUNE_LR),
                 loss='categorical_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.Precision(), 
                         tf.keras.metrics.Recall()])
    
    # Continue training
    callbacks_fine_tune = get_callbacks('vgg16_finetuned')
    history_fine_tune = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=Config.EPOCHS_FINE_TUNE,
        callbacks=callbacks_fine_tune
    )
    
    # Plot fine-tuning history
    plot_training_history(history_fine_tune, 'Fine-tuning')
    
    # Save final model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'{Config.MODEL_SAVE_PATH}/vgg16_lung_cancer_{timestamp}.keras'
    model.save(model_path)
    print(f'\n💾 Model saved to {model_path}')
    
    # Save class names
    class_names_path = f'{Config.MODEL_SAVE_PATH}/class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f'✓ Class names saved to {class_names_path}')
    
    # Evaluate on all datasets
    print('\n📊 Final Evaluation:')
    train_results = evaluate_model(model, train_dataset, class_names, 'Training')
    val_results = evaluate_model(model, validation_dataset, class_names, 'Validation')
    test_results = evaluate_model(model, test_dataset, class_names, 'Test')
    
    # Save evaluation results
    results = {
        'train_accuracy': train_results['accuracy'],
        'val_accuracy': val_results['accuracy'],
        'test_accuracy': test_results['accuracy'],
        'timestamp': timestamp
    }
    
    results_path = f'{Config.RESULTS_DIR}/evaluation_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'\n✅ Evaluation results saved to {results_path}')
    
    print('\n' + '='*60)
    print(' ✅ TRAINING COMPLETE!')
    print('='*60)

# =====================================================================
# PREDICTION PIPELINE
# =====================================================================
def prediction_pipeline(model_path, image_path_or_dir, class_names_path=None):
    """Load model and predict on new images"""
    print('='*60)
    print(' LUNG CANCER DETECTION - PREDICTION')
    print('='*60)
    
    # Load model
    print(f'\n📥 Loading model from {model_path}...')
    model = load_model(model_path)
    
    # Load class names
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    else:
        # Try to find class names in model directory
        class_names_file = os.path.join(os.path.dirname(model_path), 'class_names.json')
        if os.path.exists(class_names_file):
            with open(class_names_file, 'r') as f:
                class_names = json.load(f)
        else:
            class_names = ['Normal', 'Cancer']  # Default for binary
            print('⚠️ Class names file not found. Using default: [Normal, Cancer]')
    
    # Check if path is file or directory
    if os.path.isfile(image_path_or_dir):
        predict_single_image(model, image_path_or_dir, class_names)
    elif os.path.isdir(image_path_or_dir):
        predict_batch(model, image_path_or_dir, class_names)
    else:
        print('❌ Invalid path!')

# =====================================================================
# MENU SYSTEM
# =====================================================================
def display_menu():
    """Display main menu"""
    print('\n' + '='*60)
    print(' LUNG CANCER DETECTION SYSTEM')
    print('='*60)
    print('1. Train New Model')
    print('2. Predict Single Image')
    print('3. Predict Batch Images')
    print('4. Evaluate Existing Model')
    print('5. Exit')
    print('='*60)
    
    choice = input('Enter your choice (1-5): ')
    return choice

if __name__ == "__main__":
    # Check if base directory exists
    if not os.path.exists(Config.BASE_DIR):
        print(f'❌ Error: Directory "{Config.BASE_DIR}" not found!')
        print(f'Please create a directory named "{Config.BASE_DIR}" with subdirectories for each class.')
        print('Example structure:')
        print('  abc/')
        print('    ├── normal/')
        print('    │   ├── image1.jpg')
        print('    │   └── image2.jpg')
        print('    └── cancer/')
        print('        ├── image1.jpg')
        print('        └── image2.jpg')
        exit(1)
    
    # Main loop
    while True:
        choice = display_menu()
        
        if choice == '1':
            main()
        elif choice == '2':
            model_path = input('Enter model path (.keras file): ')
            image_path = input('Enter image path: ')
            prediction_pipeline(model_path, image_path)
        elif choice == '3':
            model_path = input('Enter model path (.keras file): ')
            image_dir = input('Enter image directory path: ')
            prediction_pipeline(model_path, image_dir)
        elif choice == '4':
            model_path = input('Enter model path (.keras file): ')
            model = load_model(model_path)
            # Evaluate on test set if available
            if os.path.exists(Config.BASE_DIR):
                class_names, num_classes = get_class_info(Config.BASE_DIR)
                _, _, test_dataset = load_dataset(Config.BASE_DIR, Config.BATCH_SIZE, 
                                                 subset='validation', validation_split=0.5)
                evaluate_model(model, test_dataset, class_names, 'Test')
        elif choice == '5':
            print('\n👋 Thank you for using the Lung Cancer Detection System!')
            break
        else:
            print('\n❌ Invalid choice! Please try again.')
