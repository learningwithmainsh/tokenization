# NLP Pipeline with Docker

This project demonstrates a simple NLP pipeline for tokenization, stop-word removal, TF-IDF computation, and sentiment analysis using NLTK and Scikit-learn. The pipeline is containerized using Docker.

## 🚀 Getting Started

### 1. Clone the repository
```bash
# Clone this repository to your local machine
git clone https://github.com/learningwithmainsh/tokenization.git
cd tokenization
```

### 2. Build the Docker image
```bash
# Build the Docker image
docker build -t nlp_pipeline .
```

### 3. Run the container
```bash
# Run the container
docker run -d --name nlp_pipeline_container nlp_pipeline
```

### 4. Check logs (optional)
```bash
# View logs to check output
docker logs nlp_pipeline_container
```

### 5. Access the running container
```bash
# If you need to access the container's shell
docker exec -it nlp_pipeline_container /bin/bash
```

## 📦 Dockerfile Overview

The Dockerfile contains the following steps:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY nlp_pipeline.py nlp_pipeline.py

CMD ["python", "nlp_pipeline.py"]
```

## 📚 Requirements

The following dependencies are installed inside the Docker container:

- `nltk==3.8.1`
- `scikit-learn==1.3.0`
- `numpy==1.23.5`

These versions are specified in `requirements.txt`.

## 🛑 Stopping and Removing the Container

To stop the running container:
```bash
docker stop nlp_pipeline_container
```

To remove the container:
```bash
docker rm nlp_pipeline_container
```

## ✅ Testing

Once the container is running, you can review the output of tokenization, TF-IDF computation, and sentiment analysis directly in the logs.

## 📊 NLP Pipeline Flow

1. **Tokenization**: Splitting text into individual words or tokens.
2. **Stop-word Removal**: Removing common words (like 'the', 'is', etc.) to focus on meaningful words.
3. **TF-IDF Computation**: Calculating Term Frequency-Inverse Document Frequency scores to highlight important words.
4. **Sentiment Analysis**: Analyzing text polarity — positive, negative, or neutral.

## 🤝 Contributing

Contributions are welcome! If you have ideas to enhance the NLP pipeline or add new features, feel free to fork the repo and submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## 📞 Support

If you have any questions or suggestions, feel free to reach out!

- [LinkedIn](https://www.linkedin.com/in/learningwithmanish)
- [WhatsApp](https://wa.me/918765368754)
- [Skype](live:.cid.16ae1ff3196c4f4e)
- [Medium](https://medium.com/@mnshkmrpnd)
- [Threads](https://www.threads.net/@learningwithmanish)
- [Instagram](https://www.instagram.com/learningwithmanish/?hl=en)
- [Facebook](https://www.facebook.com/pandey.manish.106)
- [DockerHub](https://hub.docker.com/u/manishgenius)

---

### Author 👤
**Manish Pandey**  
Passionate about DevOps, Cloud Engineering, and NLP. Let’s build something amazing together!

