"""
A computer vision-based system for classifying waste types from image data
to support smart waste management systems.
"""

import os
import io
import time
import json
import base64
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0, MobileNetV3Large, ResNet50V2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import flask
from flask import Flask, request, jsonify
import threading
import argparse
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("waste_classifier.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("WasteClassifier")

# Define waste categories - these should align with your training data
WASTE_CATEGORIES = [
    'cardboard',
    'glass',
    'metal',
    'paper',
    'plastic',
    'trash',  # non-recyclable
    'organic',
    'electronic',
    'hazardous',
    'textile'
]

class WasteClassifier:
    """
    AI model for classifying waste from images.
    """
    def __init__(self, model_path=None, model_type='efficientnet', input_size=(224, 224),
                confidence_threshold=0.60, enable_gpu=True):
        """
        Initialize the waste classifier.
        
        Parameters:
        -----------
        model_path : str or None
            Path to pre-trained model file
        model_type : str
            Base model architecture ('efficientnet', 'mobilenet', 'resnet')
        input_size : tuple
            Input image size (height, width)
        confidence_threshold : float
            Minimum confidence level for classifications
        enable_gpu : bool
            Whether to enable GPU acceleration
        """
        self.model_path = model_path
        self.model_type = model_type
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.enable_gpu = enable_gpu
        self.model = None
        self.preprocess_func = None
        
        # Set up GPU configuration
        self._setup_gpu(enable_gpu)
        
        # Load model
        self._load_model()
        
        logger.info(f"Initialized waste classifier with {model_type} model")
    
    def _setup_gpu(self, enable_gpu):
        """Configure TensorFlow to use GPU if available and requested."""
        if enable_gpu:
            # Allow memory growth to avoid taking all GPU memory
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU acceleration enabled. Found {len(gpus)} GPU(s).")
                except RuntimeError as e:
                    logger.error(f"Error setting up GPU: {str(e)}")
            else:
                logger.warning("GPU acceleration requested but no GPUs found.")
        else:
            # Disable GPU
            tf.config.set_visible_devices([], 'GPU')
            logger.info("GPU acceleration disabled.")
    
    def _load_model(self):
        """Load the classification model."""
        # Check if model file exists
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
                
                # Determine preprocessing function based on loaded model
                # This is a simplification - in practice, you might need a more robust method
                if 'efficientnet' in self.model_path.lower():
                    self.preprocess_func = efficientnet_preprocess
                    self.model_type = 'efficientnet'
                elif 'mobilenet' in self.model_path.lower():
                    self.preprocess_func = mobilenet_preprocess
                    self.model_type = 'mobilenet'
                elif 'resnet' in self.model_path.lower():
                    self.preprocess_func = resnet_preprocess
                    self.model_type = 'resnet'
                else:
                    # Default to EfficientNet preprocessing
                    self.preprocess_func = efficientnet_preprocess
                
                return
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
        
        # If no model file or loading failed, create a new model
        logger.info(f"Creating new {self.model_type} model")
        
        # Select base model and preprocessing function
        if self.model_type == 'efficientnet':
            base_model = EfficientNetB0(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.input_size, 3)
            )
            self.preprocess_func = efficientnet_preprocess
        elif self.model_type == 'mobilenet':
            base_model = MobileNetV3Large(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.input_size, 3)
            )
            self.preprocess_func = mobilenet_preprocess
        elif self.model_type == 'resnet':
            base_model = ResNet50V2(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.input_size, 3)
            )
            self.preprocess_func = resnet_preprocess
        else:
            logger.warning(f"Unknown model type '{self.model_type}'. Using EfficientNet.")
            base_model = EfficientNetB0(
                weights='imagenet', 
                include_top=False, 
                input_shape=(*self.input_size, 3)
            )
            self.preprocess_func = efficientnet_preprocess
            self.model_type = 'efficientnet'
        
        # Add classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(WASTE_CATEGORIES), activation='softmax')(x)
        
        # Create the model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Created new {self.model_type} model with {len(WASTE_CATEGORIES)} waste categories")
    
    def preprocess_image(self, img):
        """
        Preprocess image for the model.
        
        Parameters:
        -----------
        img : PIL.Image or numpy.ndarray
            Input image
            
        Returns:
        --------
        numpy.ndarray : Preprocessed image
        """
        # Convert to PIL Image if numpy array
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # Resize image
        img = img.resize(self.input_size)
        
        # Convert to numpy array
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply model-specific preprocessing
        processed_img = self.preprocess_func(img_array)
        
        return processed_img
    
    def classify(self, img):
        """
        Classify waste from an image.
        
        Parameters:
        -----------
        img : PIL.Image, numpy.ndarray, or str
            Input image, numpy array, or path to image file
            
        Returns:
        --------
        dict : Classification results including category, confidence, and all scores
        """
        # Load image if path is provided
        if isinstance(img, str) and os.path.exists(img):
            try:
                img = Image.open(img).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image from {img}: {str(e)}")
                return {"error": f"Could not load image: {str(e)}"}
        
        # Ensure image is in correct format
        if not isinstance(img, (Image.Image, np.ndarray)):
            logger.error(f"Unsupported image type: {type(img)}")
            return {"error": "Unsupported image type"}
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image(img)
            
            # Make prediction
            predictions = self.model.predict(processed_img, verbose=0)[0]
            
            # Get top prediction
            top_idx = np.argmax(predictions)
            top_category = WASTE_CATEGORIES[top_idx]
            top_confidence = float(predictions[top_idx])
            
            # Create results dictionary
            all_scores = {category: float(score) for category, score 
                         in zip(WASTE_CATEGORIES, predictions)}
            
            results = {
                "category": top_category,
                "confidence": top_confidence,
                "all_scores": all_scores,
                "is_confident": top_confidence >= self.confidence_threshold,
                "timestamp": datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during classification: {str(e)}")
            return {"error": f"Classification failed: {str(e)}"}
    
    def batch_classify(self, images):
        """
        Classify multiple images at once.
        
        Parameters:
        -----------
        images : list
            List of images (PIL.Image, numpy.ndarray, or str paths)
            
        Returns:
        --------
        list : List of classification results
        """
        results = []
        
        for img in images:
            result = self.classify(img)
            results.append(result)
        
        return results
    
    def train(self, train_dir, validation_dir=None, epochs=10, batch_size=32,
              fine_tune=True, fine_tune_epochs=5, augmentation=True,
              early_stopping=True, save_path=None):
        """
        Train the waste classification model.
        
        Parameters:
        -----------
        train_dir : str
            Path to training data directory (with subdirectories for each category)
        validation_dir : str or None
            Path to validation data directory (with subdirectories for each category)
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        fine_tune : bool
            Whether to fine-tune some top layers of the base model
        fine_tune_epochs : int
            Number of fine-tuning epochs
        augmentation : bool
            Whether to use data augmentation
        early_stopping : bool
            Whether to use early stopping
        save_path : str or None
            Path to save trained model
            
        Returns:
        --------
        dict : Training history
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        
        logger.info(f"Starting model training from {train_dir}")
        
        # Create data generators with augmentation if requested
        if augmentation:
            train_datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                preprocessing_function=self.preprocess_func
            )
        else:
            train_datagen = ImageDataGenerator(
                preprocessing_function=self.preprocess_func
            )
        
        # Validation data generator (no augmentation)
        val_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_func
        )
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        # Check if categories match expected
        categories = list(train_generator.class_indices.keys())
        if set(categories) != set(WASTE_CATEGORIES):
            logger.warning(f"Training data categories {categories} don't match expected categories {WASTE_CATEGORIES}")
        
        # Create validation generator if validation directory provided
        validation_generator = None
        if validation_dir and os.path.exists(validation_dir):
            validation_generator = val_datagen.flow_from_directory(
                validation_dir,
                target_size=self.input_size,
                batch_size=batch_size,
                class_mode='categorical'
            )
        
        # Prepare callbacks
        callbacks = []
        
        if early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor='val_accuracy' if validation_generator else 'accuracy',
                    patience=5,
                    restore_best_weights=True
                )
            )
        
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_accuracy' if validation_generator else 'accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Initial training with frozen base model
        logger.info("Starting initial training phase...")
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        initial_history = {
            'accuracy': history.history['accuracy'],
            'loss': history.history['loss']
        }
        
        if validation_generator:
            initial_history['val_accuracy'] = history.history['val_accuracy']
            initial_history['val_loss'] = history.history['val_loss']
        
        fine_tune_history = None
        
        # Fine-tuning phase
        if fine_tune:
            logger.info("Starting fine-tuning phase...")
            
            # Unfreeze some top layers of the base model
            # The choice of how many layers to unfreeze depends on the model
            if self.model_type == 'efficientnet':
                for layer in self.model.layers[-20:]:
                    layer.trainable = True
            elif self.model_type == 'mobilenet':
                for layer in self.model.layers[-15:]:
                    layer.trainable = True
            elif self.model_type == 'resnet':
                for layer in self.model.layers[-30:]:
                    layer.trainable = True
            
            # Recompile model with lower learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train with fine-tuning
            ft_history = self.model.fit(
                train_generator,
                epochs=fine_tune_epochs,
                validation_data=validation_generator,
                callbacks=callbacks,
                verbose=1
            )
            
            fine_tune_history = {
                'accuracy': ft_history.history['accuracy'],
                'loss': ft_history.history['loss']
            }
            
            if validation_generator:
                fine_tune_history['val_accuracy'] = ft_history.history['val_accuracy']
                fine_tune_history['val_loss'] = ft_history.history['val_loss']
        
        # Save the final model if not using ModelCheckpoint
        if save_path and not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
            try:
                self.model.save(save_path)
                logger.info(f"Saved trained model to {save_path}")
                self.model_path = save_path
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
        
        # Update model path
        if save_path:
            self.model_path = save_path
        
        # Combine histories
        full_history = {
            'initial_training': initial_history,
            'fine_tuning': fine_tune_history
        }
        
        logger.info("Model training completed")
        return full_history
    
    def evaluate(self, test_dir, batch_size=32, visualize=True, output_path=None):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        test_dir : str
            Path to test data directory (with subdirectories for each category)
        batch_size : int
            Batch size for evaluation
        visualize : bool
            Whether to create visualization of results
        output_path : str or None
            Path to save evaluation results
            
        Returns:
        --------
        dict : Evaluation results
        """
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        logger.info(f"Evaluating model on test data from {test_dir}")
        
        # Create data generator
        test_datagen = ImageDataGenerator(
            preprocessing_function=self.preprocess_func
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.input_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Get class indices
        class_indices = test_generator.class_indices
        classes = {v: k for k, v in class_indices.items()}
        
        # Evaluate model
        logger.info("Running evaluation...")
        evaluation = self.model.evaluate(test_generator, verbose=1)
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true classes
        true_classes = test_generator.classes
        
        # Generate classification report
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=list(class_indices.keys()),
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Visualization
        if visualize:
            self._visualize_evaluation(cm, list(class_indices.keys()), 
                                      report, evaluation, output_path)
        
        # Save results if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            results = {
                'accuracy': float(evaluation[1]),
                'loss': float(evaluation[0]),
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'class_indices': class_indices
            }
            
            try:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=4)
                logger.info(f"Saved evaluation results to {output_path}")
            except Exception as e:
                logger.error(f"Error saving evaluation results: {str(e)}")
        
        logger.info(f"Evaluation completed. Accuracy: {evaluation[1]:.4f}")
        
        return {
            'accuracy': float(evaluation[1]),
            'loss': float(evaluation[0]),
            'classification_report': report,
            'confusion_matrix': cm,
            'class_indices': class_indices
        }
    
    def _visualize_evaluation(self, cm, class_names, report, evaluation, output_path=None):
        """Create visualization of evaluation results."""
        plt.figure(figsize=(12, 10))
        
        # Plot confusion matrix
        plt.subplot(2, 1, 1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right')
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Plot category-wise metrics
        plt.subplot(2, 1, 2)
        
        categories = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        precisions = [report[cat]['precision'] for cat in categories]
        recalls = [report[cat]['recall'] for cat in categories]
        f1_scores = [report[cat]['f1-score'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.25
        
        plt.bar(x - width, precisions, width, label='Precision')
        plt.bar(x, recalls, width, label='Recall')
        plt.bar(x + width, f1_scores, width, label='F1-Score')
        
        plt.xlabel('Categories')
        plt.ylabel('Scores')
        plt.title(f'Category-wise Metrics (Overall Accuracy: {evaluation[1]:.4f})')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.legend()
        
        plt.tight_layout()
        
        # Save if output path provided
        if output_path:
            viz_path = output_path.replace('.json', '_viz.png')
            plt.savefig(viz_path)
            logger.info(f"Saved evaluation visualization to {viz_path}")
        
        plt.show()
    
    def save_model(self, path=None):
        """
        Save the model to file.
        
        Parameters:
        -----------
        path : str or None
            Path to save model (uses self.model_path if None)
        """
        save_path = path or self.model_path
        if not save_path:
            logger.warning("No save path specified")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            self.model.save(save_path)
            logger.info(f"Saved model to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def set_confidence_threshold(self, threshold):
        """
        Set the confidence threshold for classifications.
        
        Parameters:
        -----------
        threshold : float
            New confidence threshold (0-1)
        """
        if 0 <= threshold <= 1:
            self.confidence_threshold = threshold
            logger.info(f"Set confidence threshold to {threshold}")
        else:
            logger.error(f"Invalid confidence threshold: {threshold}. Must be between 0 and 1.")


class WasteClassifierAPI:
    """
    Flask API for the waste classification service.
    """
    def __init__(self, model_path=None, host='0.0.0.0', port=5000, debug=False,
                model_type='efficientnet', confidence_threshold=0.6):
        """
        Initialize the waste classifier API.
        
        Parameters:
        -----------
        model_path : str or None
            Path to pre-trained model file
        host : str
            Host address to listen on
        port : int
            Port to listen on
        debug : bool
            Whether to run Flask in debug mode
        model_type : str
            Type of model to use
        confidence_threshold : float
            Confidence threshold for classifications
        """
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.debug = debug
        
        # Initialize classifier
        self.classifier = WasteClassifier(
            model_path=model_path,
            model_type=model_type,
            confidence_threshold=confidence_threshold
        )
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_classifications': 0,
            'failed_classifications': 0,
            'category_counts': {category: 0 for category in WASTE_CATEGORIES},
            'start_time': datetime.now(),
            'last_request_time': None
        }
        
        # Set up routes
        self._setup_routes()
        
        logger.info(f"Initialized Waste Classifier API on {host}:{port}")
    
    def _setup_routes(self):
        """Set up API routes."""
        @self.app.route('/api/classify', methods=['POST'])
        def classify_waste():
            """Classify waste from uploaded image."""
            self.stats['total_requests'] += 1
            self.stats['last_request_time'] = datetime.now()
            
            # Check if request has file
            if 'image' not in request.files:
                self.stats['failed_classifications'] += 1
                return jsonify({
                    'error': 'No image file provided',
                    'status': 'error'
                }), 400
            
            try:
                # Get image file
                img_file = request.files['image']
                
                # Read image
                img = Image.open(img_file.stream).convert('RGB')
                
                # Classify image
                result = self.classifier.classify(img)
                
                # Check for error
                if 'error' in result:
                    self.stats['failed_classifications'] += 1
                    return jsonify({
                        'error': result['error'],
                        'status': 'error'
                    }), 500
                
                # Update statistics
                self.stats['successful_classifications'] += 1
                self.stats['category_counts'][result['category']] += 1
                
                # Add request ID
                result['request_id'] = str(uuid.uuid4())
                result['status'] = 'success'
                
                return jsonify(result), 200
                
            except Exception as e:
                logger.error(f"Error processing classification request: {str(e)}")
                self.stats['failed_classifications'] += 1
                return jsonify({
                    'error': f"Classification failed: {str(e)}",
                    'status': 'error'
                }), 500
        
        @self.app.route('/api/classify/base64', methods=['POST'])
        def classify_waste_base64():
            """Classify waste from base64-encoded image."""
            self.stats['total_requests'] += 1
            self.stats['last_request_time'] = datetime.now()
            
            # Check if request has JSON
            if not request.is_json:
                self.stats['failed_classifications'] += 1
                return jsonify({
                    'error': 'Request must be JSON',
                    'status': 'error'
                }), 400
            
            # Get JSON data
            data = request.get_json()
            
            # Check if data has image
            if 'image' not in data:
                self.stats['failed_classifications'] += 1
                return jsonify({
                    'error': 'No image data provided',
                    'status': 'error'
                }), 400
            
            try:
                # Get base64 image data
                img_data = data['image']
                
                # Check for data URI format and remove header if present
                if img_data.startswith('data:image'):
                    img_data = img_data.split(',')[1]
                
                # Decode base64
                img_bytes = base64.b64decode(img_data)
                
                # Convert to image
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                
                # Classify image
                result = self.classifier.classify(img)
                
                # Check for error
                if 'error' in result:
                    self.stats['failed_classifications'] += 1
                    return jsonify({
                        'error': result['error'],
                        'status': 'error'
                    }), 500
                
                # Update statistics
                self.stats['successful_classifications'] += 1
                self.stats['category_counts'][result['category']] += 1
                
                # Add request ID
                result['request_id'] = str(uuid.uuid4())
                result['status'] = 'success'
                
                return jsonify(result), 200
                
            except Exception as e:
                logger.error(f"Error processing base64 classification request: {str(e)}")
                self.stats['failed_classifications'] += 1
                return jsonify({
                    'error': f"Classification failed: {str(e)}",
                    'status': 'error'
                }), 500
        
        @self.app.route('/api/stats', methods=['GET'])
        def get_stats():
            """Get API usage statistics."""
            # Calculate uptime
            uptime = datetime.now() - self.stats['start_time']
            uptime_str = str(uptime).split('.')[0]  # Remove microseconds
            
            # Calculate success rate
            total = self.stats['total_requests']
            success_rate = (self.stats['successful_classifications'] / total * 100) if total > 0 else 0
            
            # Get top categories
            category_counts = self.stats['category_counts']
            top_categories = sorted(
                [(cat, count) for cat, count in category_counts.items() if count > 0],
                key=lambda x: x[1],
                reverse=True
            )[:5]  # Top 5
            
            stats_response = {
                'total_requests': self.stats['total_requests'],
                'successful_classifications': self.stats['successful_classifications'],
                'failed_classifications': self.stats['failed_classifications'],
                'success_rate': f"{success_rate:.2f}%",
                'uptime': uptime_str,
                'top_categories': [
                    {'category': cat, 'count': count}
                    for cat, count in top_categories
                ],
                'start_time': self.stats['start_time'].isoformat(),
                'last_request_time': (self.stats['last_request_time'].isoformat() 
                                    if self.stats['last_request_time'] else None)
            }
            
            return jsonify(stats_response), 200
        
        @self.app.route('/api/categories', methods=['GET'])
        def get_categories():
            """Get list of supported waste categories."""
            return jsonify({
                'categories': WASTE_CATEGORIES,
                'count': len(WASTE_CATEGORIES)
            }), 200
        
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'model_type': self.classifier.model_type,
                'time': datetime.now().isoformat()
            }), 200
    
    def run(self, threaded=True):
        """
        Run the API server.
        
        Parameters:
        -----------
        threaded : bool
            Whether to run Flask in threaded mode
        """
        logger.info(f"Starting API server on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug, threaded=threaded)

def run_classifier_api(model_path=None, host='0.0.0.0', port=5000, debug=False,
                   model_type='efficientnet', confidence_threshold=0.6):
    """
    Run the waste classifier API.
    
    Parameters:
    -----------
    model_path : str or None
        Path to pre-trained model file
    host : str
        Host address to listen on
    port : int
        Port to listen on
    debug : bool
        Whether to run Flask in debug mode
    model_type : str
        Type of model to use
    confidence_threshold : float
        Confidence threshold for classifications
        
    Returns:
    --------
    WasteClassifierAPI instance
    """
    api = WasteClassifierAPI(
        model_path=model_path,
        host=host,
        port=port,
        debug=debug,
        model_type=model_type,
        confidence_threshold=confidence_threshold
    )
    api.run()
    return api

def run_standalone_classifier(image_path, model_path=None, model_type='efficientnet',
                            confidence_threshold=0.6):
    """
    Run the waste classifier on a single image.
    
    Parameters:
    -----------
    image_path : str
        Path to image file
    model_path : str or None
        Path to pre-trained model file
    model_type : str
        Type of model to use
    confidence_threshold : float
        Confidence threshold for classifications
        
    Returns:
    --------
    dict : Classification results
    """
    # Initialize classifier
    classifier = WasteClassifier(
        model_path=model_path,
        model_type=model_type,
        confidence_threshold=confidence_threshold
    )
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return {"error": "Image file not found"}
    
    # Classify image
    result = classifier.classify(image_path)
    
    # Print results
    if 'error' in result:
        logger.error(f"Classification failed: {result['error']}")
        print(f"Classification failed: {result['error']}")
    else:
        logger.info(f"Classification successful: {result['category']} ({result['confidence']:.2f})")
        print(f"\nClassification Results:")
        print(f"  Category: {result['category']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Confident: {'Yes' if result['is_confident'] else 'No'}")
        print("\nAll Categories:")
        for category, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {score:.4f}")
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Waste Classification API")
    
    # API mode arguments
    parser.add_argument("--api", action="store_true", help="Run as API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    # Classifier arguments
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--model-type", type=str, default="efficientnet", 
                       choices=["efficientnet", "mobilenet", "resnet"], 
                       help="Type of model architecture")
    parser.add_argument("--threshold", type=float, default=0.6, 
                       help="Confidence threshold (0-1)")
    
    # Training arguments
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--train-dir", type=str, help="Path to training data directory")
    parser.add_argument("--val-dir", type=str, help="Path to validation data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save-path", type=str, help="Path to save trained model")
    
    # Evaluation arguments
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--test-dir", type=str, help="Path to test data directory")
    
    # Image classification argument
    parser.add_argument("--image", type=str, help="Path to image file to classify")
    
    args = parser.parse_args()
    
    # Run in requested mode
    if args.api:
        # Run as API server
        run_classifier_api(
            model_path=args.model,
            host=args.host,
            port=args.port,
            debug=args.debug,
            model_type=args.model_type,
            confidence_threshold=args.threshold
        )
    elif args.train:
        # Train model
        if not args.train_dir:
            logger.error("Training directory not specified")
            print("Please specify training directory with --train-dir")
            sys.exit(1)
        
        # Initialize classifier
        classifier = WasteClassifier(
            model_path=args.model,
            model_type=args.model_type,
            confidence_threshold=args.threshold
        )
        
        # Train model
        classifier.train(
            train_dir=args.train_dir,
            validation_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_path=args.save_path or args.model
        )
    elif args.evaluate:
        # Evaluate model
        if not args.test_dir:
            logger.error("Test directory not specified")
            print("Please specify test directory with --test-dir")
            sys.exit(1)
        
        # Initialize classifier
        classifier = WasteClassifier(
            model_path=args.model,
            model_type=args.model_type,
            confidence_threshold=args.threshold
        )
        
        # Evaluate model
        classifier.evaluate(
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            visualize=True
        )
    elif args.image:
        # Classify single image
        run_standalone_classifier(
            image_path=args.image,
            model_path=args.model,
            model_type=args.model_type,
            confidence_threshold=args.threshold
        )
    else:
        # No mode specified, print help
        parser.print_help()

"""
SUMMARY:
--------
This module implements an AI-powered waste classification system that leverages
computer vision to identify waste types from images. Key components include:

1. WasteClassifier: Core class for waste image classification using deep learning
   - Supports multiple model architectures (EfficientNet, MobileNet, ResNet)
   - Includes training, evaluation, and inference capabilities
   - Handles preprocessing and augmentation for robust performance

2. WasteClassifierAPI: Flask-based API for deploying the classifier as a service
   - RESTful endpoints for image classification (file upload and base64)
   - Statistics tracking and health monitoring
   - Supports containerization for scalable deployment

The system supports 10 waste categories and provides confidence scores for
classification results. It can be run as a standalone classifier or as an API
service, making it suitable for integration with waste management systems.

TODO:
-----
- Implement transfer learning for fine-tuning on specific waste streams
- Add support for object detection to handle multiple waste items in one image
- Implement automated model retraining with new data
- Add support for video stream processing for real-time classification
- Create more extensive data augmentation for better generalization
- Implement model explainability to visualize what the model is focusing on
- Add multi-language support for API responses
- Create containerized deployment with Docker for easier scaling
"""
