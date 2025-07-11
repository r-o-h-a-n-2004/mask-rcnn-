import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.datasets import cifar10  # Using CIFAR10 for demo

# Configuration
class Config:
    NAME = "MaskRCNN"
    BACKBONE = "resnet50"
    IMAGE_SHAPE = (128, 128, 3)  # Reduced for demo
    NUM_CLASSES = 10  # CIFAR10 classes
    RPN_ANCHOR_SCALES = (8, 16, 32)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 4
    POOL_SIZE = (7, 7)
    MASK_POOL_SIZE = (14, 14)
    MASK_SHAPE = (28, 28)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 10
    DETECTION_MAX_INSTANCES = 10
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

config = Config()

# 1. Backbone Network
def build_backbone():
    """Build ResNet50 backbone with Feature Pyramid Network (FPN)"""
    # Input
    input_image = layers.Input(shape=config.IMAGE_SHAPE, name="input_image")
    
    # ResNet50
    resnet = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=input_image)
    
    # Feature Pyramid Network
    C2, C3, C4, C5 = [
        resnet.get_layer(layer_name).output
        for layer_name in ["conv2_block3_out", "conv3_block4_out", 
                           "conv4_block6_out", "conv5_block3_out"]
    ]
    
    # Top-down FPN
    P5 = layers.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P4 = layers.Add(name="fpn_p4add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        layers.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    ])
    P3 = layers.Add(name="fpn_p3add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        layers.Conv2D(256, (1, 1), name='fpn_c3p3')(C3)
    ])
    P2 = layers.Add(name="fpn_p2add")([
        layers.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        layers.Conv2D(256, (1, 1), name='fpn_c2p2')(C2)
    ])
    
    # Add 3x3 conv to all P layers
    P2 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = layers.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(P5)
    
    return models.Model(input_image, [P2, P3, P4, P5], name="backbone")

# 2. Region Proposal Network (RPN)
def build_rpn(feature_map):
    """Build RPN for a single feature map"""
    shared = layers.Conv2D(512, (3, 3), padding='same', activation='relu',
                           name='rpn_conv_shared')(feature_map)
    
    # Anchor classification (bg/fg)
    x_class = layers.Conv2D(2 * len(config.RPN_ANCHOR_RATIOS), (1, 1), 
                            name='rpn_class_raw')(shared)
    rpn_class_logits = layers.Reshape(
        [-1, 2], name='rpn_class_reshape')(x_class)
    rpn_probs = layers.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)
    
    # Bounding box regression
    x_reg = layers.Conv2D(4 * len(config.RPN_ANCHOR_RATIOS), (1, 1),
                          name='rpn_bbox_pred')(shared)
    rpn_bbox = layers.Reshape(
        [-1, 4], name='rpn_bbox_reshape')(x_reg)
    
    return rpn_class_logits, rpn_probs, rpn_bbox

# 3. ROIAlign Layer
class ROIAlign(layers.Layer):
    """Implements ROIAlign layer"""
    def __init__(self, pool_size, **kwargs):
        super(ROIAlign, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        # Unpack inputs
        boxes, feature_map = inputs
        
        # Crop and resize regions
        box_indices = tf.zeros(tf.shape(boxes)[0], dtype=tf.int32)
        return tf.image.crop_and_resize(
            feature_map, boxes, box_indices, self.pool_size,
            method="bilinear", extrapolation_value=0)

    def compute_output_shape(self, input_shape):
        return (None, self.pool_size[0], self.pool_size[1], input_shape[1][3])

# 4. Detection Head
def build_fpn_classifier(feature_map, num_classes):
    """Build classification and box regression heads"""
    # Shared features
    x = layers.TimeDistributed(layers.Conv2D(1024, (config.POOL_SIZE[0], config.POOL_SIZE[1]), padding="valid"),
                           name="mrcnn_class_conv1")(feature_map)
    x = layers.TimeDistributed(layers.BatchNormalization()), name='mrcnn_class_bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.TimeDistributed(layers.Conv2D(1024, (1, 1)),
                           name="mrcnn_class_conv2")(x)
    x = layers.TimeDistributed(layers.BatchNormalization()), name='mrcnn_class_bn2')(x)
    x = layers.Activation('relu')(x)
    
    shared = layers.TimeDistributed(layers.Flatten(), name="mrcnn_class_flatten")(x)
    
    # Classifier
    mrcnn_class_logits = layers.TimeDistributed(
        layers.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = layers.TimeDistributed(
        layers.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)
    
    # BBox regression
    mrcnn_bbox = layers.TimeDistributed(
        layers.Dense(4 * num_classes), name='mrcnn_bbox_fc')(shared)
    mrcnn_bbox = layers.Reshape((-1, num_classes, 4), name="mrcnn_bbox")(mrcnn_bbox)
    
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

# 5. Mask Head
def build_mask_head(feature_map, num_classes):
    """Build mask segmentation head"""
    x = layers.TimeDistributed(
        layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(feature_map)
    x = layers.TimeDistributed(layers.BatchNormalization()), name='mrcnn_mask_bn1')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(x)
    x = layers.TimeDistributed(layers.BatchNormalization()), name='mrcnn_mask_bn2')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(x)
    x = layers.TimeDistributed(layers.BatchNormalization()), name='mrcnn_mask_bn3')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.TimeDistributed(
        layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(x)
    x = layers.TimeDistributed(layers.BatchNormalization()), name='mrcnn_mask_bn4')(x)
    x = layers.Activation('relu')(x)
    
    x = layers.TimeDistributed(
        layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"), 
        name="mrcnn_mask_deconv")(x)
    mrcnn_mask = layers.TimeDistributed(
        layers.Conv2D(num_classes, (1, 1), activation="sigmoid"), 
        name="mrcnn_mask")(x)
    
    return mrcnn_mask

# 6. Build Complete Mask R-CNN
def build_mask_rcnn(config):
    # Inputs
    input_image = layers.Input(shape=config.IMAGE_SHAPE, name="input_image")
    input_gt_boxes = layers.Input(shape=(config.MAX_GT_INSTANCES, 4), name="input_gt_boxes")
    
    # Backbone
    backbone = build_backbone()
    P2, P3, P4, P5 = backbone(input_image)
    
    # RPN on each feature map
    rpn_feature_maps = [P2, P3, P4, P5]
    rpn_class_logits = []
    rpn_probs = []
    rpn_bbox = []
    
    for p in rpn_feature_maps:
        logits, probs, bbox = build_rpn(p)
        rpn_class_logits.append(logits)
        rpn_probs.append(probs)
        rpn_bbox.append(bbox)
    
    # Combine RPN outputs
    rpn_class_logits = layers.Concatenate(axis=1, name="rpn_class_logits")(rpn_class_logits)
    rpn_probs = layers.Concatenate(axis=1, name="rpn_class")(rpn_probs)
    rpn_bbox = layers.Concatenate(axis=1, name="rpn_bbox")(rpn_bbox)
    
    # Generate ROIs (simplified for demo)
    rois = layers.Lambda(lambda x: generate_rois(*x), name="ROI")(
        [rpn_probs, rpn_bbox, input_gt_boxes])
    
    # ROIAlign (using P2 for demo)
    roi_align = ROIAlign(config.POOL_SIZE)
    feature_map = roi_align([rois, P2])
    
    # Detection Head
    mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = build_fpn_classifier(feature_map, config.NUM_CLASSES)
    
    # Mask Head
    mrcnn_mask = build_mask_head(feature_map, config.NUM_CLASSES)
    
    return models.Model(
        [input_image, input_gt_boxes],
        [rpn_class_logits, rpn_probs, rpn_bbox, rois, 
         mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_mask],
        name='mask_rcnn'
    )

# ROI Generator (simplified)
def generate_rois(rpn_probs, rpn_bbox, gt_boxes):
    # In real implementation: apply NMS, select top proposals
    return tf.random.uniform((config.TRAIN_ROIS_PER_IMAGE, 4), 0, 1, dtype=tf.float32)

# 7. Build and Compile Model
model = build_mask_rcnn(config)
model.compile(optimizer='adam', loss='mse')  # Simplified for demo
model.summary()

# 8. Prepare Dataset (CIFAR10)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = np.array([cv2.resize(img, config.IMAGE_SHAPE[:2]) for img in x_train])
x_test = np.array([cv2.resize(img, config.IMAGE_SHAPE[:2]) for img in x_test])

# Dummy GT boxes (for demo)
gt_boxes = np.zeros((len(x_train), config.MAX_GT_INSTANCES, 4))

# 9. Train the Model
print("Training... (this may take a while)")
model.fit(
    [x_train[:1000], gt_boxes[:1000]],  # Using subset for demo
    np.zeros((1000, 8)),  # Dummy targets
    epochs=1, 
    batch_size=1
)

# 10. Inference Function
def detect_objects(image, model):
    # Preprocess
    image = cv2.resize(image, config.IMAGE_SHAPE[:2])
    image = np.expand_dims(image, 0)
    
    # Dummy GT boxes
    gt_boxes = np.zeros((1, config.MAX_GT_INSTANCES, 4))
    
    # Run inference
    _, _, _, rois, _, mrcnn_probs, mrcnn_bbox, _ = model.predict([image, gt_boxes])
    
    # Process results
    class_ids = np.argmax(mrcnn_probs[0], axis=1)
    scores = np.max(mrcnn_probs[0], axis=1)
    
    # Filter detections
    keep = scores > config.DETECTION_MIN_CONFIDENCE
    class_ids = class_ids[keep]
    scores = scores[keep]
    boxes = rois[0][keep]
    
    return boxes, class_ids, scores

# 11. Run Detection and Show Results
# Select test image
image = x_test[0]
boxes, class_ids, scores = detect_objects(image, model)

# Visualize
plt.figure(figsize=(12, 8))
plt.imshow(image.astype('uint8'))
ax = plt.gca()

# CIFAR10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for i, box in enumerate(boxes):
    y1, x1, y2, x2 = box * np.array([image.shape[0], image.shape[1], 
                                    image.shape[0], image.shape[1]])
    width, height = x2 - x1, y2 - y1
    ax.add_patch(plt.Rectangle((x1, y1), width, height, 
                               fill=False, color='red', linewidth=2))
    label = f"{class_names[class_ids[i]]}: {scores[i]:.2f}"
    ax.text(x1, y1 - 10, label, color='white', fontsize=12, 
            bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')
plt.title('Mask R-CNN Detection Results')
plt.show()

# Print confidence scores
print("\nDetection Confidence Scores:")
for i, score in enumerate(scores):
    print(f"- {class_names[class_ids[i]]}: {score:.4f}")