# 🧠 Multimodal Hate Speech Detection in Memes (Subtask A)

This project tackles **Subtask A** of a shared task on hate speech detection, focusing on the **binary classification** of memes into `Hate` or `No Hate`. It leverages a **multimodal deep learning pipeline** integrating both **textual** and **visual** cues from meme images.

## 🎯 Task Description

Detect whether a meme contains **hate speech** using:

* Text extracted from the meme (via OCR)
* Image features (via CNN)
* Textual semantics (via BERT)

## 📁 Project Structure

```
.
├── shared-task-a-initial.ipynb    # Baseline multimodal model (OCR + BERT + ResNet)
├── subtask-a-deep.ipynb           # Advanced model with transformer and training enhancements
├── data/                          # Folder with training/eval images and CSVs
├── outputs/                       # Predictions, logs, evaluation metrics
├── README.md
```

## 🧰 Technologies Used

* **Torch** for deep learning
* **Transformers** (HuggingFace BERT)
* **Torchvision** (ResNet18/50 for image embeddings)
* **EasyOCR** for text extraction from images
* **Scikit-learn** for metrics
* **Pandas/Numpy** for data handling

## 🧠 Model Pipeline

### Step 1: Text Extraction

* Use **EasyOCR** to extract embedded text from memes.

### Step 2: Text Embedding

* Use **BERT tokenizer and model** to generate text embeddings from OCR output.

### Step 3: Image Embedding

* Use **ResNet** pretrained CNN to generate image feature vectors.

### Step 4: Fusion and Classification

* Concatenate text and image embeddings.
* Pass through a classifier (MLP or attention-based) for hate speech prediction.

## 🧪 Evaluation

* Accuracy, Precision, Recall, F1 Score
* Classification report on test set

## 🧾 Dataset

* Dataset from the shared task organizer includes:

  * Meme images
  * Labels in CSV (hate or no hate)
  * Unique ID matching image and label

## 🧠 Enhancements in `subtask-a-deep.ipynb`

* Early stopping and scheduler
* Use of `AutoModel` for flexibility in text model
* Model saving/loading
* Training time monitoring

## 📚 References

* HuggingFace Transformers
* Torchvision Models
* EasyOCR
* Shared Task Description: *Multimodal Hate Speech Classification*

