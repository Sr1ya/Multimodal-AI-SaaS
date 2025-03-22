# 🔥 **Multimodal Emotion Recognition Using MELD Dataset on AWS Sagemaker**

## 🚀 **Project Overview**
This project implements a multimodal emotion recognition model using the **MELD dataset**, combining **text, audio, and visual features** through **late fusion**. The model is deployed and trained on **AWS Sagemaker** with **TensorBoard logs** for monitoring.

---

## 🛠️ **Model Introduction**
![Model Introduction](.images/Model Intro.png)   

**Explanation:**  
The model takes **three different modalities** as input:  
- **Text features** extracted using a pre-trained transformer model.  
- **Audio features** processed with a convolutional neural network (CNN).  
- **Visual features** extracted using a pre-trained image model.  
The outputs from these models are **fused using late fusion** to enhance accuracy.

---

## 📊 **Dataset Preparation**
![Dataset Preparation](.images/Dataset-Prep.png)  

**Explanation:**  
The **MELD dataset** contains:  
- **Text**: Dialogue transcripts.  
- **Audio**: Speech recordings.  
- **Visual**: Facial expressions from videos.  
The dataset is preprocessed by:  
1. Tokenizing text using `Hugging Face` transformers.  
2. Extracting **MFCC features** from audio.  
3. Using **ResNet or EfficientNet** for visual embeddings.  
4. Synchronizing modalities using time-aligned frames.

---

## ⚙️ **Model Architecture**
![Model Architecture](.images/Architecture)  

**Explanation:**  
The architecture consists of:  
- **Text Encoder:** Transformer-based model for text representation.  
- **Audio Encoder:** CNN or LSTM for speech features.  
- **Visual Encoder:** Pretrained CNN for visual embeddings.  
- **Late Fusion Layer:** Combines all three modalities using a **concatenation + dense layer**.  
- **Classification Layer:** Applies softmax for multi-class emotion prediction.

---

## 🔥 **Model Fusion**
![Model Fusion](.images/Model Fusion)  

**Explanation:**  
**Late Fusion** is performed by:  
1. Extracting separate embeddings from the **text, audio, and visual encoders**.  
2. Concatenating the embeddings.  
3. Passing the combined representation through a **fully connected dense layer**.  
4. Applying softmax for emotion classification.  

---

## 🔎 **Detailed Model**
![Detailed Model](.images/Detailed Model)  

**Explanation:**  
The detailed model flow includes:  
- Preprocessing pipelines for text, audio, and visual data.  
- Separate encoders for feature extraction.  
- Late fusion to combine features.  
- Output layers with **categorical cross-entropy loss** for multi-class classification.

---

## ☁️ **AWS Architecture**
![AWS Architecture](.images/AWS Architecture)  

**Explanation:**  
The training and retraining process uses **AWS infrastructure**:  
- **EC2 Instances:** For model training.  
- **S3 Buckets:** For dataset storage and model artifacts.  
- **TensorBoard Logs:** For tracking training metrics.  
- **Lambda Functions:** For triggering retraining on new data.  
- **SageMaker:** For deploying and serving the model.

---

## 🔥 **Training and Retraining Process**
![Training and Retraining](.images/training)  

**Explanation:**  
1. **Initial Training:**  
   - Model is trained on the MELD dataset using TensorFlow/PyTorch.  
   - TensorBoard logs are generated and stored in AWS.  
2. **Retraining Workflow:**  
   - New data is added to the S3 bucket.  
   - AWS Lambda triggers a retraining job.  
   - The updated model is deployed automatically.  

---


Features:

- 🎥 Video sentiment analysis
- 📺 Video frame extraction
- 🎙️ Audio feature extraction
- 📝 Text embedding with BERT
- 🔗 Multimodal fusion
- 📊 Emotion and sentiment classification
- 🚀 Model training and evaluation
- 📈 TensorBoard logging
- 🚀 AWS S3 for video storage
- 🤖 AWS SageMaker endpoint integration
- 🔐 User authentication with Auth.js
- 🔑 API key management
- 📊 Usage quota tracking
- 📈 Real-time analysis results
- 🎨 Modern UI with Tailwind CSS

## Setup

Follow these steps to install and set up the project.

### Clone the Repository

```bash
git clone https://github.com/Andreaswt/ai-video-sentiment-model.git
```

### Navigate to the Project Directory

```bash
cd ai-video-sentiment-model
```

### Install Python

Download and install Python if not already installed. Use the link below for guidance on installation:
[Python Download](https://www.python.org/downloads/)

### Install Dependencies

```bash
pip install -r training/requirements.txt
```

### Download the Dataset

Visit the following link to download the MELD dataset:
[MELD Dataset](https://affective-meld.github.io)

Extract the dataset and place it in the `dataset` directory.

PS: learn more about state-of-the-art model in the following [Emotion Recognition Benchmark for the MELD dataset ](https://paperswithcode.com/sota/emotion-recognition-in-conversation-on-meld).

### Start Training Job

Follow these steps to train the model in a training job using AWS SageMaker:

1. Request a quota increase for an instance for training job usage for SageMaker - e.g. ml.g5.xlarge

2. Put the dataset in an S3 bucket

3. Create a role with Policies

- AmazonSageMakerFullAccess
- Access to S3 bucket with dataset

```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "VisualEditor0",
			"Effect": "Allow",
			"Action": [
				"s3:PutObject",
				"s3:GetObject",
				"s3:ListBucket",
				"s3:DeleteObject"
			],
			"Resource": [
				"arn:aws:s3:::your-bucket-name",
				"arn:aws:s3:::your-bucket-name/*"
			]
		}
	]
}
```

4. Run the file locally, to start the training job.

```bash
python train_sagemaker.py
```

### Deploy Endpoint

Follow these steps to deploy the model as an endpoint using AWS SageMaker:

1. Create a deployment role in AWS with permissions

- AmazonSageMakerFullAccess
- CloudWatchLogsFullAccess

```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "VisualEditor0",
			"Effect": "Allow",
			"Action": [
				"s3:PutObject",
				"s3:GetObject",
				"s3:ListBucket",
				"s3:DeleteObject"
			],
			"Resource": [
				"arn:aws:s3:::your-bucket-name",
				"arn:aws:s3:::your-bucket-name/*"
			]
		}
	]
}
```

2. Put your model file in an S3 bucket

3. Deploy the endpoint by runnin the file locally:

```bash
python deployment/deploy_endpoint.py
```

### Invoke Endpoint

1. Create a user in IAM with permissions

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject"
            ],
            "Resource": [
                "arn:aws:s3:::sentiment-analysis-saas/inference/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:us-east-1:784061079855:endpoint/sentiment-analysis-endpoint"
            ]
        }
    ]
}
```

2. Use the user to invoke endpoint. E.g. use [this NPM library](https://www.npmjs.com/package/@aws-sdk/client-sagemaker-runtime) for invoking from JavaScript:

### Access TensorBoard

1. Download logs to local machine:
   `aws s3 sync s3://your-bucket-name/tensorboard ./tensorboard_logs`

2. Start tensorboard server
   `tensorboard --logdir tensorboard_logs`

3. Open your browser and visit:
   [http://localhost:6006](http://localhost:6006)


## Setup

Follow these steps to install and set up the SaaS project:

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-video-sentiment-saas.git
cd ai-video-sentiment-saas
```

2. Install dependencies

```
npm install
```

3. Configure environment variables in .env:

```
DATABASE_URL="your-database-url"
AUTH_SECRET="your-auth-secret"
AWS_REGION="your-aws-region"
AWS_ACCESS_KEY_ID="your-access-key"
AWS_SECRET_ACCESS_KEY="your-secret-key"
```

4. Initialize the database:

```
npm run db:generate
npm run db:push
```

## Running the app

### Development

```
npm run dev
```

### Production

```
npm run build
npm start
```
