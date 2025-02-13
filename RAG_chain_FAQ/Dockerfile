FROM python:3.12.7

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install nbconvert

RUN jupyter nbconvert --to script RAG.ipynb

CMD ["python", "RAG.py"]